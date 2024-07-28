#![allow(warnings)]
#![warn(clippy::semicolon_if_nothing_returned)]
#![cfg_attr(not(test), warn(unused_crate_dependencies, unused_extern_crates))]

use acvm::FieldElement;
use brillig::BinaryFieldOp;
use brillig::BinaryIntOp;
use brillig::HeapArray;
use brillig::MemoryAddress;
use brillig::Opcode;
use brillig::ValueOrArray;
use env_logger::Env;
use inkwell::intrinsics::Intrinsic;
use inkwell::values::AnyValue;
use inkwell::values::PointerValue;
use log::warn;
use std::collections::HashMap;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::env;
use std::fmt::Pointer;
use std::fs;
use std::hash::Hash;
use std::mem;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;
use ark_ff::BigInt;

use inkwell::context::Context;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::module::Module;
use inkwell::types::{BasicTypeEnum, IntType};
use inkwell::values::{BasicValueEnum, IntValue};
use inkwell::AddressSpace;
use inkwell::IntPredicate;

mod instructions;
mod opcodes;
mod transpile;
mod transpile_contract;
mod utils;

use transpile_contract::CompiledArtifact;

type BrilligOpcode = brillig::Opcode<FieldElement>;

fn main() {
    let args: Vec<String> = env::args().collect();
    let in_artifact_path = &args[1];
    let out_artifact_path = &args[2];
    let json_parse_error = format!("Unable to parse json for: {in_artifact_path}");

    // Parse original (pre-transpile) contract.
    let json = fs::read_to_string(Path::new(in_artifact_path))
        .expect(&format!("Unable to read file: {in_artifact_path}"));

    // Backup the output file if it already exists.
    if Path::new(out_artifact_path).exists() {
        std::fs::copy(
            Path::new(out_artifact_path),
            Path::new(&(out_artifact_path.clone() + ".bak")),
        )
        .expect(&format!("Unable to backup file: {out_artifact_path}"));
    }

    let load_start = Instant::now();
    let artifact: CompiledArtifact = serde_json::from_str(&json).expect(&json_parse_error);
    eprintln!("Artifact load took: {:?}", load_start.elapsed());

    for function in &artifact.bytecode.unconstrained_functions {
        generate_llvm_ir(&function.bytecode);
        // println!("");
        // for opcode in &function.bytecode {
        //     println!("{:?}", opcode);
        // }
    }
}

fn field_element_to_u64_limbs(fe: FieldElement) -> [u64; 4] {
        let arkfe: ark_bn254::Fr = fe.into_repr();
        let big_int: BigInt<4> = arkfe.into();
        big_int.0
}

fn disassemble_brillig(opcodes: &Vec<BrilligOpcode>) -> BTreeMap<usize, BTreeMap<usize, Vec<BrilligOpcode>>> {
    // Create the map of locations to their corresponding opcodes
    let mut opcode_map: BTreeMap<usize, BTreeMap<usize, Vec<BrilligOpcode>>> = BTreeMap::new();

    // Init the vector of function locations with initial entrypoint (address 0).
    let mut function_locations: Vec<usize> = vec![0];
    // Extract unique call locations.
    let call_locations: HashSet<usize> = opcodes.iter()
        .filter_map(|opcode| {
            if let BrilligOpcode::Call { location } = opcode {
                Some(*location)
            } else {
                None
            }
        })
        .collect();
    // Add them to the list of function locations.
    function_locations.extend(call_locations);

    // For each function entry location.
    // - Scan forward until we find a jump, return or stop.
    // - Construct the basic block and add it to the set, keyed on its location.
    // - Add any newly discovered basic blocks (jump targets / fallthroughs) to a process list.
    // - Keep going until process list is empty.
    for location in function_locations {
        let mut basic_blocks: BTreeMap<usize, Vec<BrilligOpcode>> = BTreeMap::new();

        // The list of locations to process.
        // We keep processing until the list is empty.
        // Initialize it with the entrypoint location of the function.
        let mut to_process: Vec<usize> = vec![location];

        // Scan forward until we find a jump, return or stop.
        while !to_process.is_empty() {
            // Pop the first entry off the list.
            // This is the start location of the basic block.
            let start_location = to_process.remove(0);

            for (index, opcode) in opcodes[start_location..].iter().enumerate() {
                match opcode {
                    // BrilligOpcode::Call { location: _ } => {
                    //     let block_opcodes = opcodes[start_location..start_location + index + 1].to_vec();
                    //     basic_blocks.insert(start_location, block_opcodes);
                    //     let after_loc = start_location + index + 1;
                    //     if (!basic_blocks.contains_key(&after_loc)) {
                    //         to_process.push(after_loc);
                    //     }
                    //     break;
                    // }
                    BrilligOpcode::Stop { return_data_offset: _, return_data_size: _ } |
                    BrilligOpcode::Trap { revert_data: _ } |
                    BrilligOpcode::Return => {
                        // We've found the end of a basic block.
                        // For stop and return, we're at the end of function execution.
                        // We expect to_process to be empty and this to be the final basic block.
                        // assert!(to_process.is_empty());
                        let block_opcodes = opcodes[start_location..start_location + index + 1].to_vec();
                        basic_blocks.insert(start_location, block_opcodes);
                        break;
                    }
                    BrilligOpcode::Jump { location: jump_loc } => {
                        // We've found the end of a basic block.
                        // This is an unconditional jump, so signals the start of another basic block at the target.
                        // Add the basic block and add the jump location to the to_process list.
                        let block_opcodes = opcodes[start_location..start_location + index + 1].to_vec();
                        basic_blocks.insert(start_location, block_opcodes);
                        if (!basic_blocks.contains_key(jump_loc)) {
                            to_process.push(*jump_loc);
                        }
                        break;
                    }
                    BrilligOpcode::JumpIf { location: jump_loc, condition: _ } |
                    BrilligOpcode::JumpIfNot { location: jump_loc, condition: _ } => {
                        // We've found the end of a basic block.
                        // This is a conditional jump, so signals the start of two new basic blocks.
                        // One is the target of the jump, the other is the instruction after this one.
                        let block_opcodes = opcodes[start_location..start_location + index + 1].to_vec();
                        basic_blocks.insert(start_location, block_opcodes);
                        if (!basic_blocks.contains_key(jump_loc)) {
                            to_process.push(*jump_loc);
                        }
                        let after_loc = start_location + index + 1;
                        if (!basic_blocks.contains_key(&after_loc)) {
                            to_process.push(after_loc);
                        }
                        break;
                    }
                    _ => {}
                }
            }
        }

        opcode_map.insert(location, basic_blocks);
    }

    opcode_map
}

fn generate_llvm_ir(opcodes: &Vec<BrilligOpcode>) {
    let context = Context::create();
    let module = context.create_module("brillig");
    let builder = context.create_builder();
    let i256_type = context.custom_width_int_type(256);
    let i128_type = context.i128_type();
    let i64_type = context.i64_type();
    let i32_type = context.i32_type();
    let i8_type = context.i8_type();
    let i1_type = context.bool_type();
    let i8_ptr_type = i8_type.ptr_type(AddressSpace::default());
    let i64_ptr_type = i64_type.ptr_type(AddressSpace::default());
    let i256_ptr_type = i256_type.ptr_type(AddressSpace::default());

    // Declare external functions
    let bn254_fr_add = module.add_function("bn254_fr_add", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_sub = module.add_function("bn254_fr_sub", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_mul = module.add_function("bn254_fr_mul", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_eql = module.add_function("bn254_fr_eql", i1_type.fn_type(&[i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bb_printf_func = module.add_function("bb_printf", i32_type.fn_type(&[i8_ptr_type.into()], true), None);
    let printf_func = module.add_function("printf", i32_type.fn_type(&[i8_ptr_type.into()], true), None);
    let to_radix_func = module.add_function("to_radix", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_type.into(), i64_type.into()], false), None);

    // Function signature for main
    let fn_type = i32_type.fn_type(&[], false);
    let main_function = module.add_function("main", fn_type, None);
    let entry_block = context.append_basic_block(main_function, "entry");
    builder.position_at_end(entry_block);

    // Define global calldata
    let calldata_type = i256_type.array_type(2);
    let calldata = module.add_global(calldata_type, Some(AddressSpace::default()), "calldata");
    calldata.set_alignment(32);
    calldata.set_initializer(&i256_type.const_array(&[
        i256_type.const_int(2, false),
        i256_type.const_int(1, false),
    ]));

    // Define memory.
    let memory_type = i256_type.array_type(2051);
    let memory_global = module.add_global(memory_type, Some(AddressSpace::default()), "memory");
    memory_global.set_alignment(32);
    memory_global.set_initializer(&i256_type.const_array(&vec![i256_type.const_int(0, false); 2051]));
    let memory = memory_global.as_pointer_value();

    // Define 4MB heap.
    let heap_type = i8_type.array_type(1024*1024*4);
    let heap_global = module.add_global(heap_type, Some(AddressSpace::default()), "heap");
    heap_global.set_alignment(32);
    heap_global.set_initializer(&i8_type.const_array(&vec![i8_type.const_int(0, false); 1024*1024*4]));
    let heap = heap_global.as_pointer_value();

    macro_rules! get_calldata_at_index {
        ($index:expr) => {
            unsafe {
                builder.build_gep(i256_type, calldata.as_pointer_value(), &[i32_type.const_int($index as u64, false)], "elem_ptr")
            }.unwrap()
        };
    }

    macro_rules! get_memory_at_index {
        ($index:expr) => {
            unsafe {
                builder.build_gep(i256_type, memory, &[i32_type.const_int($index as u64, false)], "elem_ptr")
            }.unwrap()
        };
    }

    macro_rules! get_memory_at_index_ {
        ($index:expr) => {
            unsafe {
                builder.build_gep(i256_type, memory, &[$index], "elem_ptr")
            }.unwrap()
        };
    }

    // let mut call_locs: Vec<u64> = vec![];
    // let mut current_function = main_function;
    // let mut function_stack: Vec<inkwell::values::FunctionValue> = vec![main_function];

    // Print opcode counts.
    let mut counts: HashMap<String, usize> = HashMap::new();
    for item in opcodes {
        let variant_name = format!("{:?}", item);
        let variant_key = variant_name.split_whitespace().next().unwrap().to_string();
        let counter = counts.entry(variant_key).or_insert(0);
        *counter += 1;
    }
    for (key, value) in &counts {
        eprintln!("{:?} occurs {} times", key, value);
    }

    let dis_start = Instant::now();
    let opcode_map = disassemble_brillig(opcodes);
    eprintln!("Dissassembly took: {:?}", dis_start.elapsed());

    // for (location, ops) in opcode_map.clone() {
    //     println!("Location: {}", location);
    //     for op in ops {
    //         println!("{:?}", op);
    //     }
    // }

    let build_start = Instant::now();
    // Pre-add all functions to the module, so we can reference them in calls.
    for (function_location, blocks) in &opcode_map {
        let function_name = format!("function_at_{}", function_location);
        let function_type = context.void_type().fn_type(&[], false);
        module.add_function(&function_name, function_type, None);
    }

    // module.print_to_stderr();
    // return;
    for (function_location, blocks) in &opcode_map {
        let function_name = format!("function_at_{}", function_location);
        let function = module.get_function(&function_name).unwrap();

        // Create a map of basic blocks.
        let mut block_map: HashMap<usize, BasicBlock> = HashMap::new();
        for (block_location, opcodes) in blocks {
            let block_name = format!("block_at_{}", block_location);
            block_map.insert(*block_location, context.append_basic_block(function, &block_name));
        }

        for (block_location, opcodes) in blocks {
            let block = block_map[block_location];
            builder.position_at_end(block);
            for (opcode_index, opcode) in opcodes.iter().enumerate() {
                let before_count = block.get_instructions().count();
                // let comment_str = context.metadata_string(&format!("{:?}", opcode));
                // let comment_node = context.metadata_node(&[comment_str.into()]);

                match opcode {
                    BrilligOpcode::Const { destination, bit_size, value } => {
                        let dest_index = destination.0;
                        // let int_type = context.custom_width_int_type(*bit_size);
                        let limbs = field_element_to_u64_limbs(*value);
                        // let const_val = int_type.const_int_arbitrary_precision(&limbs);
                        let const_val = i256_type.const_int_arbitrary_precision(&limbs);
                        let dest_ptr = get_memory_at_index!(dest_index);
                        builder.build_store(dest_ptr, const_val).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::CalldataCopy { destination_address, size, offset } => {
                        for i in 0..*size {
                            let addr = destination_address.0 + i;
                            let src_ptr = get_calldata_at_index!(i + offset);
                            let dest_ptr = get_memory_at_index!(addr);
                            let value = builder.build_load(i256_type, src_ptr, "loadtmp").unwrap();
                            builder.build_store(dest_ptr, value).unwrap().set_alignment(32);
                        }
                    }
                    BrilligOpcode::Mov { destination, source } => {
                        let src_index = source.0;
                        let dest_index = destination.0;
                        let src_ptr = get_memory_at_index!(src_index);
                        let dest_ptr = get_memory_at_index!(dest_index);
                        let value = builder.build_load(i256_type, src_ptr, "loadtmp").unwrap();
                        builder.build_store(dest_ptr, value).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::Cast { destination, source, bit_size } => {
                        // TODO: Some kind of range check or something!?
                        // Currently just same as Mov.
                        let src_index = source.0;
                        let dest_index = destination.0;
                        let src_ptr = get_memory_at_index!(src_index);
                        let dest_ptr = get_memory_at_index!(dest_index);
                        let value = builder.build_load(i256_type, src_ptr, "loadtmp").unwrap();
                        builder.build_store(dest_ptr, value).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::Store { destination_pointer, source } => {
                        let src_index = source.0;
                        let src_ptr = get_memory_at_index!(src_index);
                        let value = builder.build_load(i256_type, src_ptr, "loadtmp").unwrap();

                        let dest_ptr_index = destination_pointer.0;
                        let dest_ptr_ptr = get_memory_at_index!(dest_ptr_index);
                        let dest_ptr = builder.build_load(i256_type, dest_ptr_ptr, "loadtmp").unwrap();
                        let dest_gep = get_memory_at_index_!(dest_ptr.into_int_value());

                        builder.build_store(dest_gep, value).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::Load { destination, source_pointer } => {
                        let dest_index = destination.0;
                        let dest_ptr = get_memory_at_index!(dest_index);

                        let src_ptr_index = source_pointer.0;
                        let src_ptr_ptr = get_memory_at_index!(src_ptr_index);
                        let src_ptr = builder.build_load(i256_type, src_ptr_ptr, "loadtmp").unwrap();
                        let src_gep = get_memory_at_index_!(src_ptr.into_int_value());
                        let value = builder.build_load(i256_type, src_gep, "loadtmp").unwrap();

                        builder.build_store(dest_ptr, value).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::BinaryFieldOp { destination, op, lhs, rhs } => {
                        let lhs_index = lhs.0;
                        let rhs_index = rhs.0;
                        let result_index = destination.0;

                        let lhs_ptr = get_memory_at_index!(lhs_index);
                        let rhs_ptr = get_memory_at_index!(rhs_index);
                        let result_ptr = get_memory_at_index!(result_index);

                        match op {
                            BinaryFieldOp::Add => {
                                builder.build_call(bn254_fr_add, &[lhs_ptr.into(), rhs_ptr.into(), result_ptr.into()], "add_call");
                            }
                            BinaryFieldOp::Sub => {
                                builder.build_call(bn254_fr_sub, &[lhs_ptr.into(), rhs_ptr.into(), result_ptr.into()], "sub_call");
                            }
                            BinaryFieldOp::Mul => {
                                builder.build_call(bn254_fr_mul, &[lhs_ptr.into(), rhs_ptr.into(), result_ptr.into()], "mul_call");
                            }
                            BinaryFieldOp::Equals => {
                                let lhs_val = builder.build_load(i256_type, lhs_ptr, "load_lhs").unwrap().into_int_value();
                                let rhs_val = builder.build_load(i256_type, rhs_ptr, "load_rhs").unwrap().into_int_value();
                                builder.build_int_compare(IntPredicate::EQ, lhs_val, rhs_val, "eq").unwrap();
                                // let result = builder.build_call(bn254_fr_eql, &[lhs_ptr.into(), rhs_ptr.into()], "eql_call").unwrap();
                                // builder.build_store(result_ptr, result.try_as_basic_value().unwrap_left());
                            }
                            BinaryFieldOp::LessThanEquals => {
                                let lhs_val = builder.build_load(i256_type, lhs_ptr, "load_lhs").unwrap().into_int_value();
                                let rhs_val = builder.build_load(i256_type, rhs_ptr, "load_rhs").unwrap().into_int_value();
                                builder.build_int_compare(IntPredicate::ULE, lhs_val, rhs_val, "ule").unwrap();
                            }
                            _ => unimplemented!("Unimplemented BinaryFieldOp variant: {:?}", op),
                        }
                    }
                    BrilligOpcode::BinaryIntOp { destination, op, bit_size, lhs, rhs } => {
                        let lhs_index = lhs.0;
                        let rhs_index = rhs.0;
                        let result_index = destination.0;

                        let lhs_ptr = get_memory_at_index!(lhs_index);
                        let rhs_ptr = get_memory_at_index!(rhs_index);
                        let result_ptr = get_memory_at_index!(result_index);

                        // TODO: Handle correct bit_size.
                        let lhs_val = builder.build_load(i256_type, lhs_ptr, "load_lhs").unwrap().into_int_value();
                        let rhs_val = builder.build_load(i256_type, rhs_ptr, "load_rhs").unwrap().into_int_value();

                        let value = match op {
                            BinaryIntOp::Add => {
                                builder.build_int_add(lhs_val, rhs_val, "add").unwrap()
                            }
                            BinaryIntOp::Sub => {
                                builder.build_int_sub(lhs_val, rhs_val, "sub").unwrap()
                            }
                            BinaryIntOp::Mul => {
                                builder.build_int_mul(lhs_val, rhs_val, "mul").unwrap()
                            }
                            BinaryIntOp::Div => {
                                builder.build_int_unsigned_div(lhs_val, rhs_val, "udiv").unwrap()
                            }
                            BinaryIntOp::LessThan => {
                                builder.build_int_compare(IntPredicate::ULT, lhs_val, rhs_val, "ult").unwrap()
                            }
                            BinaryIntOp::LessThanEquals => {
                                builder.build_int_compare(IntPredicate::ULE, lhs_val, rhs_val, "ule").unwrap()
                            }
                            BinaryIntOp::Equals => {
                                builder.build_int_compare(IntPredicate::EQ, lhs_val, rhs_val, "eq").unwrap()
                            }
                            BinaryIntOp::Shl => {
                                builder.build_left_shift(lhs_val, rhs_val, "shl").unwrap()
                            }
                            BinaryIntOp::Shr => {
                                builder.build_right_shift(lhs_val, rhs_val, false, "shr").unwrap()
                            }
                            BinaryIntOp::Or => {
                                builder.build_or(lhs_val, rhs_val, "or").unwrap()
                            }
                            BinaryIntOp::And => {
                                builder.build_and(lhs_val, rhs_val, "and").unwrap()
                            }
                            _ => unimplemented!("Unimplemented BinaryIntOp variant: {:?}", op)
                        };
                        builder.build_store(result_ptr, value);
                    }
                    BrilligOpcode::Call { location } => {
                        let callee_name = format!("function_at_{}", location);
                        if let Some(callee) = module.get_function(&callee_name) {
                            builder.build_call(callee, &[], "calltmp");
                        } else {
                            panic!("Function {} not found", callee_name);
                        }
                    }
                    BrilligOpcode::Jump { location } => {
                        builder.build_unconditional_branch(block_map[location]);
                    }
                    BrilligOpcode::JumpIf { condition, location } => {
                        let cond_index = condition.0;
                        let cond_ptr = get_memory_at_index!(cond_index);
                        let cond_val = builder.build_load(i1_type, cond_ptr, "load_cond").unwrap().into_int_value();
                        builder.build_conditional_branch(cond_val, block_map[location], block_map[&(block_location+opcode_index+1)]);
                    }
                    BrilligOpcode::JumpIfNot { condition, location } => {
                        let cond_index = condition.0;
                        let cond_ptr = get_memory_at_index!(cond_index);
                        let cond_val = builder.build_load(i1_type, cond_ptr, "load_cond").unwrap().into_int_value();
                        builder.build_conditional_branch(cond_val.const_not(), block_map[location], block_map[&(block_location+opcode_index+1)]);
                    }
                    BrilligOpcode::Trap { revert_data } => {
                        let value = revert_data;
                        let lhs_ptr = get_memory_at_index!(value.pointer.0);
                        let ptr_val = builder.build_load(i256_type, lhs_ptr, "load").unwrap();
                        let dest_gep = get_memory_at_index_!(ptr_val.into_int_value());
                        builder.build_call(
                            bb_printf_func,
                            &[ dest_gep.into() ],
                            "printf_call",
                        );
                        let trap_intrinsic = Intrinsic::find("llvm.trap").unwrap();
                        let trap_function = trap_intrinsic.get_declaration(&module, &[]).unwrap();
                        builder.build_call(trap_function, &[], "trap_call");
                        builder.build_unreachable();
                    }
                    BrilligOpcode::Stop { return_data_offset, return_data_size } => {
                        builder.build_return(None);
                    }
                    BrilligOpcode::Return => {
                        builder.build_return(None);
                    }
                    BrilligOpcode::BlackBox(op) => {
                        // TODO: IMPLEMENT!
                        match op {
                            brillig::BlackBoxOp::ToRadix { input, radix, output } => {
                                let input_ptr = get_memory_at_index!(input.0);
                                let output_ptr_ptr = get_memory_at_index!(output.pointer.0);
                                let output_ptr = builder.build_load(i256_type, output_ptr_ptr, "loadtmp").unwrap();
                                let output_gep = get_memory_at_index_!(output_ptr.into_int_value());
                                let size = i64_type.const_int(output.size as u64, false);
                                let radix = i64_type.const_int(*radix as u64, false);
                                builder.build_call(
                                    to_radix_func,
                                    &[ input_ptr.into(), output_gep.into(), size.into(), radix.into() ],
                                    "to_radix_call",
                                );
                            }
                            _ => eprintln!("Skipping BlackBox: {:?}", op)
                        }
                    }
                    BrilligOpcode::ForeignCall { function, destinations, destination_value_types, inputs, input_value_types} => {
                        match function.as_str() {
                            "print" => {
                                // eprintln!("{:?}", inputs);
                                // eprintln!("{:?}", input_value_types);
                                // eprintln!("{:?}", destinations);
                                // eprintln!("{:?}", destination_value_types);
                                let value = if let ValueOrArray::HeapArray(v) = inputs[1] { v } else { unreachable!() };
                                let lhs_ptr = get_memory_at_index!(value.pointer.0);
                                let ptr_val = builder.build_load(i256_type, lhs_ptr, "load").unwrap();
                                let dest_gep = get_memory_at_index_!(ptr_val.into_int_value());
                                builder.build_call(
                                    bb_printf_func,
                                    &[ dest_gep.into() ],
                                    "printf_call",
                                );
                            }
                            _ => eprintln!("Skipping ForeignCall: {:?}", function),
                        }
                    }
                    _ => unimplemented!("Unimplemented enum variant: {:?}", opcode),
                }

                // let inst_count = block.get_instructions().count() - before_count;
                // let opcode_str = &format!("{:?}", opcode);
                // eprintln!("{inst_count}: {opcode_str}");
            }
        }
    }

    builder.position_at_end(entry_block);
    let function = module.get_function("function_at_0").unwrap();
    builder.build_call(function, &[], "calltmp");

    let format_str = builder.build_global_string_ptr(
        "0x%016llx%016llx%016llx%016llx\n",
        "format_str",
    ).unwrap();

    // let i64_type = context.i64_type();
    let indices = [
        i64_type.const_int(0, false),
        i64_type.const_int(1, false),
        i64_type.const_int(2, false),
        i64_type.const_int(3, false),
    ];

    let base_gep = unsafe {
        builder.build_gep(i256_type, memory, &[i64_type.const_int(2050, false)], "base_elem_ptr")
    }.unwrap();

    let values: Vec<_> = indices.iter().map(|&index| {
        let elem_gep = unsafe {
            builder.build_gep(i64_type, base_gep, &[index], "elem_ptr")
        }.unwrap();
        builder.build_load(i64_type, elem_gep, "val").unwrap().into_int_value()
    }).collect();

    builder.build_call(
        printf_func,
        &[
            format_str.as_pointer_value().into(),
            values[3].into(),
            values[2].into(),
            values[1].into(),
            values[0].into()
        ],
        "printf_call",
    );
    builder.build_return(Some(&i32_type.const_int(0, false)));

    eprintln!("Build took: {:?}", build_start.elapsed());

    let write_start = Instant::now();
    print!("{}", module.print_to_string().to_string());
    module.verify().unwrap();
    eprintln!("Write took: {:?}", write_start.elapsed());
}