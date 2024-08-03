#![allow(warnings)]
#![warn(clippy::semicolon_if_nothing_returned)]
#![cfg_attr(not(test), warn(unused_crate_dependencies, unused_extern_crates))]

use acvm::FieldElement;
use brillig::BinaryFieldOp;
use brillig::BinaryIntOp;
use brillig::HeapArray;
use brillig::IntegerBitSize;
use brillig::MemoryAddress;
use brillig::Opcode;
use brillig::ValueOrArray;
use env_logger::Env;
use inkwell::intrinsics::Intrinsic;
use inkwell::values::AnyValue;
use inkwell::values::BasicValue;
use inkwell::values::PointerValue;
use log::warn;
use std::collections::HashMap;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::env;
use std::error::Error;
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
use inkwell::types::{BasicTypeEnum, IntType, VectorType};
use inkwell::values::{BasicValueEnum, IntValue};
use inkwell::AddressSpace;
use inkwell::IntPredicate;

type BrilligOpcode = brillig::Opcode<FieldElement>;

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

pub fn generate_llvm_ir(opcodes: &Vec<BrilligOpcode>, calldata_fields: &Vec<FieldElement>) {
    let context = Context::create();
    let module = context.create_module("brillig");
    let builder = context.create_builder();
    let i256_type = context.custom_width_int_type(256);
    let i128_type = context.i128_type();
    let i64_type = context.i64_type();
    let i32_type = context.i32_type();
    let i16_type = context.i16_type();
    let i8_type = context.i8_type();
    let i1_type = context.bool_type();
    let i8_ptr_type = i8_type.ptr_type(AddressSpace::default());
    let i64_ptr_type = i64_type.ptr_type(AddressSpace::default());
    let i256_ptr_type = i256_type.ptr_type(AddressSpace::default());
    let v256_type = i64_type.vec_type(4);

    // Declare external functions
    let bn254_fr_add = module.add_function("bn254_fr_add", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_sub = module.add_function("bn254_fr_sub", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_mul = module.add_function("bn254_fr_mul", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_div = module.add_function("bn254_fr_div", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_eql = module.add_function("bn254_fr_eql", i1_type.fn_type(&[i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bb_printf_func = module.add_function("bb_printf", i32_type.fn_type(&[i8_ptr_type.into()], true), None);
    let print_fields_func = module.add_function("print_u256", context.void_type().fn_type(&[i64_ptr_type.into(), i64_type.into()], false), None);
    let to_radix_func = module.add_function("to_radix", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_type.into(), i64_type.into()], false), None);
    let sha256_func = module.add_function("blackbox_sha256", context.void_type().fn_type(&[i8_ptr_type.into(), i64_type.into(), i8_ptr_type.into()], false), None);

    // Function signature for main
    let fn_type = i32_type.fn_type(&[], false);
    let main_function = module.add_function("main", fn_type, None);
    let entry_block = context.append_basic_block(main_function, "entry");
    builder.position_at_end(entry_block);

    // Define global calldata
    let calldata_type = i256_type.array_type(calldata_fields.len() as u32);
    let calldata = module.add_global(calldata_type, Some(AddressSpace::default()), "calldata");
    calldata.set_alignment(32);
    let cd = calldata_fields.iter().map(|&f| i256_type.const_int_arbitrary_precision(&field_element_to_u64_limbs(f))).collect::<Vec<IntValue>>();
    calldata.set_initializer(&i256_type.const_array(&cd));

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
                        // let int_type = context.custom_width_int_type(*bit_size);
                        let limbs = field_element_to_u64_limbs(*value);
                        // let const_val = i256_type.const_int_arbitrary_precision(&limbs);
                        let const_val = VectorType::const_vector(&limbs.iter().map(|&x| i64_type.const_int(x as u64, false)).collect::<Vec<_>>());

                        let dest_index = destination.0;
                        let dest_ptr = get_memory_at_index!(dest_index);
                        builder.build_store(dest_ptr, const_val).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::CalldataCopy { destination_address, size, offset } => {
                        for i in 0..*size {
                            let addr = destination_address.0 + i;
                            let src_ptr = get_calldata_at_index!(i + offset);
                            let dest_ptr = get_memory_at_index!(addr);
                            let value = builder.build_load(v256_type, src_ptr, "cdc_val").unwrap();
                            builder.build_store(dest_ptr, value).unwrap().set_alignment(32);
                        }
                    }
                    BrilligOpcode::ConditionalMov { destination, source_a, source_b, condition } => {
                        let cond_index = condition.0;
                        let src_a_index = source_a.0;
                        let src_b_index = source_b.0;
                        let dest_index = destination.0;

                        let cond_ptr = get_memory_at_index!(cond_index);
                        let src_a_ptr = get_memory_at_index!(src_a_index);
                        let src_b_ptr = get_memory_at_index!(src_b_index);
                        let dest_ptr = get_memory_at_index!(dest_index);

                        let cond_val = builder.build_load(i256_type, cond_ptr, "cond_val").unwrap().into_int_value();
                        let src_a_val = builder.build_load(i256_type, src_a_ptr, "src_a_val").unwrap();
                        let src_b_val = builder.build_load(i256_type, src_b_ptr, "src_b_val").unwrap();

                        let zero = i256_type.const_int(0, false);
                        let cmp = builder.build_int_compare(IntPredicate::NE, cond_val, zero, "cmp").unwrap();

                        let selected_val = builder.build_select(cmp, src_a_val, src_b_val, "selected_val").unwrap();
                        builder.build_store(dest_ptr, selected_val).unwrap().set_alignment(32);
                    },
                    BrilligOpcode::Mov { destination, source } => {
                        let src_index = source.0;
                        let dest_index = destination.0;
                        let src_ptr = get_memory_at_index!(src_index);
                        let dest_ptr = get_memory_at_index!(dest_index);
                        let value = builder.build_load(v256_type, src_ptr, "mov_val").unwrap();
                        builder.build_store(dest_ptr, value).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::Cast { destination, source, bit_size } => {
                        // TODO: Some kind of range check or something!? Truncate?
                        // Currently just same as Mov.
                        let src_index = source.0;
                        let dest_index = destination.0;
                        let src_ptr = get_memory_at_index!(src_index);
                        let dest_ptr = get_memory_at_index!(dest_index);
                        let value = builder.build_load(v256_type, src_ptr, "cast_val").unwrap();
                        builder.build_store(dest_ptr, value).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::Store { destination_pointer, source } => {
                        let src_index = source.0;
                        let src_ptr = get_memory_at_index!(src_index);
                        let value = builder.build_load(v256_type, src_ptr, "store_val").unwrap();

                        let dest_ptr_index = destination_pointer.0;
                        let dest_ptr_ptr = get_memory_at_index!(dest_ptr_index);
                        let dest_ptr = builder.build_load(i32_type, dest_ptr_ptr, "store_dest").unwrap();
                        let dest_gep = get_memory_at_index_!(dest_ptr.into_int_value());

                        builder.build_store(dest_gep, value).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::Load { destination, source_pointer } => {
                        let dest_index = destination.0;
                        let dest_ptr = get_memory_at_index!(dest_index);

                        let src_ptr_index = source_pointer.0;
                        let src_ptr_ptr = get_memory_at_index!(src_ptr_index);
                        let src_ptr = builder.build_load(i256_type, src_ptr_ptr, "load_src").unwrap();
                        let src_gep = get_memory_at_index_!(src_ptr.into_int_value());
                        let value = builder.build_load(v256_type, src_gep, "load_val").unwrap();

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
                            BinaryFieldOp::Div => {
                                builder.build_call(bn254_fr_div, &[lhs_ptr.into(), rhs_ptr.into(), result_ptr.into()], "mul_call");
                            }
                            BinaryFieldOp::Equals => {
                                let lhs_val = builder.build_load(i256_type, lhs_ptr, "bfo_eq_lhs").unwrap().into_int_value();
                                let rhs_val = builder.build_load(i256_type, rhs_ptr, "bfo_eq_rhs").unwrap().into_int_value();
                                let result = builder.build_int_compare(IntPredicate::EQ, lhs_val, rhs_val, "bfo_eq").unwrap();
                                // let result = builder.build_call(bn254_fr_eql, &[lhs_ptr.into(), rhs_ptr.into()], "eql_call").unwrap();
                                builder.build_store(result_ptr, result);
                            }
                            BinaryFieldOp::LessThan => {
                                let lhs_val = builder.build_load(i256_type, lhs_ptr, "bfo_lt_lhs").unwrap().into_int_value();
                                let rhs_val = builder.build_load(i256_type, rhs_ptr, "bfo_lt_rhs").unwrap().into_int_value();
                                let result = builder.build_int_compare(IntPredicate::ULT, lhs_val, rhs_val, "bfo_lt").unwrap();
                                builder.build_store(result_ptr, result);
                            }
                            BinaryFieldOp::LessThanEquals => {
                                let lhs_val = builder.build_load(i256_type, lhs_ptr, "bfo_lte_lhs").unwrap().into_int_value();
                                let rhs_val = builder.build_load(i256_type, rhs_ptr, "bfo_lte_rhs").unwrap().into_int_value();
                                let result = builder.build_int_compare(IntPredicate::ULE, lhs_val, rhs_val, "bfo_lte").unwrap();
                                builder.build_store(result_ptr, result);
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
                        // Zero destination for < 256 bits?
                        let itype = match bit_size {
                            IntegerBitSize::U1 => i1_type,
                            IntegerBitSize::U8 => i8_type,
                            IntegerBitSize::U16 => i16_type,
                            IntegerBitSize::U32 => i32_type,
                            IntegerBitSize::U64 => i64_type,
                            IntegerBitSize::U128 => i128_type,
                            _ => unreachable!("Unsupported bit size: {bit_size}")
                        };
                        let lhs_val = builder.build_load(itype, lhs_ptr, "bio_lhs").unwrap().into_int_value();
                        let rhs_val = builder.build_load(itype, rhs_ptr, "bio_rhs").unwrap().into_int_value();

                        let value = match op {
                            BinaryIntOp::Add => {
                                builder.build_int_add(lhs_val, rhs_val, "bio_add").unwrap()
                            }
                            BinaryIntOp::Sub => {
                                builder.build_int_sub(lhs_val, rhs_val, "bio_sub").unwrap()
                            }
                            BinaryIntOp::Mul => {
                                builder.build_int_mul(lhs_val, rhs_val, "bio_mul").unwrap()
                            }
                            BinaryIntOp::Div => {
                                builder.build_int_unsigned_div(lhs_val, rhs_val, "bio_udiv").unwrap()
                            }
                            BinaryIntOp::LessThan => {
                                builder.build_int_compare(IntPredicate::ULT, lhs_val, rhs_val, "bio_ult").unwrap()
                            }
                            BinaryIntOp::LessThanEquals => {
                                builder.build_int_compare(IntPredicate::ULE, lhs_val, rhs_val, "bio_ule").unwrap()
                            }
                            BinaryIntOp::Equals => {
                                builder.build_int_compare(IntPredicate::EQ, lhs_val, rhs_val, "bio_eq").unwrap()
                            }
                            BinaryIntOp::Shl => {
                                builder.build_left_shift(lhs_val, rhs_val, "bio_shl").unwrap()
                            }
                            BinaryIntOp::Shr => {
                                // builder.build_right_shift(lhs_val, rhs_val, false, "bio_shr").unwrap()
                                let bs = itype.const_int(itype.get_bit_width().into(), false);
                                let cmp = builder.build_int_compare(IntPredicate::UGE, rhs_val, bs, "cmp_ge_bs").unwrap();
                                let zero = itype.const_int(0, false);
                                let shifted = builder.build_right_shift(lhs_val, rhs_val, false, "bio_shr").unwrap();
                                builder.build_select(cmp, zero, shifted, "select_shr").unwrap().into_int_value()
                            }
                            BinaryIntOp::Or => {
                                builder.build_or(lhs_val, rhs_val, "bio_or").unwrap()
                            }
                            BinaryIntOp::And => {
                                builder.build_and(lhs_val, rhs_val, "bio_and").unwrap()
                            }
                            BinaryIntOp::Xor => {
                                builder.build_xor(lhs_val, rhs_val, "bio_xor").unwrap()
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
                        let cond_val = builder.build_load(i1_type, cond_ptr, "ji_cond").unwrap().into_int_value();
                        builder.build_conditional_branch(cond_val, block_map[location], block_map[&(block_location+opcode_index+1)]);
                    }
                    BrilligOpcode::JumpIfNot { condition, location } => {
                        let cond_index = condition.0;
                        let cond_ptr = get_memory_at_index!(cond_index);
                        let cond_val = builder.build_load(i1_type, cond_ptr, "jin_cond").unwrap().into_int_value();
                        builder.build_conditional_branch(cond_val.const_not(), block_map[location], block_map[&(block_location+opcode_index+1)]);
                    }
                    BrilligOpcode::Trap { revert_data } => {
                        let ptr = get_memory_at_index!(revert_data.pointer.0);
                        let size = i64_type.const_int(revert_data.size as u64, false);
                        builder.build_call(print_fields_func, &[ ptr.into(), size.into()], "print_fields_call");

                        let trap_intrinsic = Intrinsic::find("llvm.trap").unwrap();
                        let trap_function = trap_intrinsic.get_declaration(&module, &[]).unwrap();
                        builder.build_call(trap_function, &[], "trap_call");
                        builder.build_unreachable();
                    }
                    BrilligOpcode::Stop { return_data_offset, return_data_size } => {
                        let ptr = get_memory_at_index!(*return_data_offset);
                        let size = i64_type.const_int(*return_data_size as u64, false);
                        builder.build_call(print_fields_func, &[ ptr.into(), size.into()], "print_fields_call");
                        builder.build_return(None);
                    }
                    BrilligOpcode::Return => {
                        builder.build_return(None);
                    }
                    BrilligOpcode::BlackBox(op) => {
                        match op {
                            brillig::BlackBoxOp::ToRadix { input, radix, output } => {
                                let input_ptr = get_memory_at_index!(input.0);
                                let output_ptr_ptr = get_memory_at_index!(output.pointer.0);
                                let output_ptr = builder.build_load(i256_type, output_ptr_ptr, "bb_tr_out_ptr").unwrap();
                                let output_gep = get_memory_at_index_!(output_ptr.into_int_value());
                                let size = i64_type.const_int(output.size as u64, false);
                                let radix = i64_type.const_int(*radix as u64, false);
                                builder.build_call(
                                    to_radix_func,
                                    &[ input_ptr.into(), output_gep.into(), size.into(), radix.into() ],
                                    "to_radix_call",
                                );
                            }
                            brillig::BlackBoxOp::Sha256 { message, output } => {
                                let message_ptr_ptr = get_memory_at_index!(message.pointer.0);
                                let message_ptr = builder.build_load(i256_type, message_ptr_ptr, "bb_sha_msg_ptr").unwrap();
                                let message_gep = get_memory_at_index_!(message_ptr.into_int_value());

                                let message_size_ptr = get_memory_at_index!(message.size.0);
                                let message_size = builder.build_load(i64_type, message_size_ptr, "bb_sha_msg_size").unwrap();

                                let output_ptr_ptr = get_memory_at_index!(output.pointer.0);
                                let output_ptr = builder.build_load(i256_type, output_ptr_ptr, "bb_sha_out_ptr").unwrap();
                                let output_gep = get_memory_at_index_!(output_ptr.into_int_value());

                                builder.build_call(
                                    sha256_func,
                                    &[ message_gep.into(), message_size.into(), output_gep.into() ],
                                    "sha256_call",
                                );
                            }
                            // _ => eprintln!("Skipping BlackBox: {:?}", op)
                            _ => unimplemented!("Unimplemented BlackBox: {:?}", op)
                        }
                    }
                    BrilligOpcode::ForeignCall { function, destinations, destination_value_types, inputs, input_value_types} => {
                        match function.as_str() {
                            "print" => {
                                // let value = if let ValueOrArray::HeapArray(v) = inputs[1] { v } else { unreachable!() };
                                match inputs[1] {
                                    ValueOrArray::HeapArray(v) =>  {
                                        let lhs_ptr = get_memory_at_index!(v.pointer.0);
                                        let ptr_val = builder.build_load(i256_type, lhs_ptr, "fc_val_ptr").unwrap();
                                        let ptr_value = get_memory_at_index_!(ptr_val.into_int_value());
                                        builder.build_call(
                                            bb_printf_func,
                                            &[ ptr_value.into() ],
                                            "printf_call",
                                        );
                                    }
                                    ValueOrArray::MemoryAddress(v) => {
                                        let ptr = get_memory_at_index!(v.0);
                                        builder.build_call(print_fields_func, &[ ptr.into(), i64_type.const_int(1, false).into()], "print_fields_call");
                                    }
                                    _ => unimplemented!("Unknown: {:?}", inputs[1]),
                                };
                            }
                            _ => eprintln!("Skipping ForeignCall: {:?}", function),
                        }
                    }
                    _ => unimplemented!("Unimplemented enum variant: {:?}", opcode),
                }

                let inst_count = block.get_instructions().count() - before_count;
                let opcode_str = &format!("{:?}", opcode);
                eprintln!("{:0>3} ({:0>2}.{:0>2}.{:0>2}): {inst_count}: {opcode_str}", block_location + opcode_index, function_location, block_location, opcode_index);
            }
        }
    }

    builder.position_at_end(entry_block);
    let function = module.get_function("function_at_0").unwrap();
    builder.build_call(function, &[], "calltmp");

    builder.build_return(Some(&i32_type.const_int(0, false)));

    eprintln!("Build took: {:?}", build_start.elapsed());

    let write_start = Instant::now();
    print!("{}", module.print_to_string().to_string());
    module.verify().unwrap();
    eprintln!("Write took: {:?}", write_start.elapsed());
}
