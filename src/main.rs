#![allow(warnings)]
#![warn(clippy::semicolon_if_nothing_returned)]
#![cfg_attr(not(test), warn(unused_crate_dependencies, unused_extern_crates))]

use acvm::FieldElement;
use brillig::BinaryFieldOp;
use brillig::BinaryIntOp;
use env_logger::Env;
use inkwell::values::PointerValue;
use log::warn;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use ark_ff::BigInt;

use inkwell::context::Context;
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

    let artifact: CompiledArtifact = serde_json::from_str(&json).expect(&json_parse_error);

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


// fn get_basic_blocks(opcodes: &Vec<BrilligOpcode>, opcodes: &Vec<BrilligOpcode>) -> HashMap<usize, Vec<BrilligOpcode>> {
// }

fn build_opcode_map(opcodes: &Vec<BrilligOpcode>) -> HashMap<usize, Vec<BrilligOpcode>> {
    // Create the map of locations to their corresponding opcodes
    let mut opcode_map: HashMap<usize, Vec<BrilligOpcode>> = HashMap::new();

    // Main function from index 0 to Stop.
    let end_index = opcodes[0..].iter().position(|opcode| matches!(opcode, BrilligOpcode::Stop {return_data_offset, return_data_size}))
        .map(|pos| pos + 1)
        .unwrap();
    opcode_map.insert(0, opcodes[0..end_index].to_vec()); // Add the main function opcodes with key 0

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

    for &location in &call_locations {
        let end_index = opcodes[location..].iter().position(|opcode| matches!(opcode, BrilligOpcode::Return))
            .map(|pos| location + pos + 1)
            .unwrap();

        let function_opcodes = opcodes[location..end_index].to_vec();

        // Scan for any Jump of JumpIf and adjust their locations.
        // For each jump outside of the current opcode slice, follow the jump, slice to the next jump, and recurse.

        opcode_map.insert(location, function_opcodes);
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
    let bn254_fq_add = module.add_function("bn254_fq_add", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fq_eql = module.add_function("bn254_fq_eql", i1_type.fn_type(&[i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let printf_func = module.add_function("printf", i32_type.fn_type(&[i8_ptr_type.into()], true), None);

    // Function signature for main
    let fn_type = i32_type.fn_type(&[], false);
    let main_function = module.add_function("main", fn_type, None);
    let entry_block = context.append_basic_block(main_function, "entry");
    builder.position_at_end(entry_block);

    // Define global calldata
    let calldata_type = i256_type.array_type(2);
    let calldata = module.add_global(calldata_type, Some(AddressSpace::default()), "calldata");
    calldata.set_initializer(&i256_type.const_array(&[
        i256_type.const_int(2, false),
        i256_type.const_int(1, false),
    ]));

    // Define memory.
    let memory_type = i256_type.array_type(2051);
    let memory_global = module.add_global(memory_type, Some(AddressSpace::default()), "memory");
    memory_global.set_initializer(&i256_type.const_array(&vec![i256_type.const_int(0, false); 2051]));
    let memory = memory_global.as_pointer_value();

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

    let opcode_map = build_opcode_map(opcodes);

    // for (location, ops) in opcode_map.clone() {
    //     println!("Location: {}", location);
    //     for op in ops {
    //         println!("{:?}", op);
    //     }
    // }

    for (function_location, opcodes) in opcode_map.clone() {
        let function_name = format!("function_at_{:x}", function_location);
        let function_type = context.void_type().fn_type(&[], false);
        module.add_function(&function_name, function_type, None);
    }

    // module.print_to_stderr();
    // return;
    for (function_location, opcodes) in opcode_map {
        let function_name = format!("function_at_{:x}", function_location);
        let function = module.get_function(&function_name).unwrap();
        let entry_block = context.append_basic_block(function, "entry");
        builder.position_at_end(entry_block);

    for opcode in opcodes {
        match opcode {
            BrilligOpcode::Const { destination, bit_size, value } => {
                let dest_index = destination.0;
                let int_type = context.custom_width_int_type(bit_size);
                let limbs = field_element_to_u64_limbs(value);
                let const_val = int_type.const_int_arbitrary_precision(&limbs);
                let dest_ptr = get_memory_at_index!(dest_index);
                builder.build_store(dest_ptr, const_val);
            }
            BrilligOpcode::CalldataCopy { destination_address, size, offset } => {
                for i in 0..size {
                    let addr = destination_address.0 + i;
                    let src_ptr = get_memory_at_index!(i + offset);
                    let dest_ptr = get_memory_at_index!(addr);
                    let value = builder.build_load(i256_type, src_ptr, "loadtmp").unwrap();
                    builder.build_store(dest_ptr, value);
                }
            }
            BrilligOpcode::Mov { destination, source } => {
                let src_index = source.0;
                let dest_index = destination.0;
                let src_ptr = get_memory_at_index!(src_index);
                let dest_ptr = get_memory_at_index!(dest_index);
                let value = builder.build_load(i256_type, src_ptr, "loadtmp").unwrap();
                builder.build_store(dest_ptr, value);
            }
            BrilligOpcode::Store { destination_pointer, source } => {
                let src_index = source.0;
                let src_ptr = get_memory_at_index!(src_index);
                let value = builder.build_load(i256_type, src_ptr, "loadtmp").unwrap();

                let dest_ptr_index = destination_pointer.0;
                let dest_ptr_ptr = get_memory_at_index!(dest_ptr_index);
                let dest_ptr = builder.build_load(i256_type, dest_ptr_ptr, "loadtmp").unwrap();
                let dest_gep = get_memory_at_index_!(dest_ptr.into_int_value());

                builder.build_store(dest_gep, value);
            }
            BrilligOpcode::Load { destination, source_pointer } => {
                let dest_index = destination.0;
                let dest_ptr = get_memory_at_index!(dest_index);

                let src_ptr_index = source_pointer.0;
                let src_ptr_ptr = get_memory_at_index!(src_ptr_index);
                let src_ptr = builder.build_load(i256_type, src_ptr_ptr, "loadtmp").unwrap();
                let src_gep = get_memory_at_index_!(src_ptr.into_int_value());
                let value = builder.build_load(i256_type, src_gep, "loadtmp").unwrap();

                builder.build_store(dest_ptr, value);
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
                        builder.build_call(bn254_fq_add, &[lhs_ptr.into(), rhs_ptr.into(), result_ptr.into()], "add_call");
                    }
                    // BinaryFieldOp::Equals => {
                    //     let eq_result = builder.build_call(bn254_fq_eql, &[lhs_ptr.into(), rhs_ptr.into()], "eq_call").try_as_basic_value().left().unwrap().into_int_value();
                    //     builder.build_store(result_ptr, builder.build_int_z_extend(eq_result, i64_type, "booltmp"));
                    // }
                    _ => unimplemented!(),
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
                let lhs_val = builder.build_load(i64_type, lhs_ptr, "load_lhs").unwrap().into_int_value();
                let rhs_val = builder.build_load(i64_type, rhs_ptr, "load_rhs").unwrap().into_int_value();

                match op {
                    BinaryIntOp::Add => {
                        let value = builder.build_int_add(lhs_val, rhs_val, "add").unwrap();
                        builder.build_store(result_ptr, value);
                    }
    //                 BinaryIntOp::Equals => {
    //                     let cmp = builder.build_int_compare(IntPredicate::EQ, lhs_val, rhs_val, "eqtmp");
    //                     builder.build_int_z_extend(cmp, i64_type, "booltmp")
    //                 }
                    _ => unimplemented!(),
                }
            }
            BrilligOpcode::Call { location } => {
                let callee_name = format!("function_at_{}", location);
                if let Some(callee) = module.get_function(&callee_name) {
                    builder.build_call(callee, &[], "calltmp");
                } else {
                    panic!("Function {} not found", callee_name);
                }
            }
    //         BrilligOpcode::JumpIf { condition, location } => {
    //             let cond_ptr = memory[condition.0].unwrap();
    //             let cond_val = builder.build_load(i64_type, cond_ptr, "load_cond").into_int_value();
    //             let zero = i64_type.const_int(0, false);
    //             let cond = builder.build_int_compare(IntPredicate::NE, cond_val, zero, "ifcond");
    //             let then_block = context.append_basic_block(function, "then");
    //             let else_block = context.append_basic_block(function, "else");
    //             builder.build_conditional_branch(cond, then_block, else_block);
    //             builder.position_at_end(then_block);
    //             builder.build_unconditional_branch(else_block);
    //             builder.position_at_end(else_block);
    //         }
    //         BrilligOpcode::Trap { revert_data } => {
    //             builder.build_call(module.get_function("llvm.trap").unwrap(), &[], "traptmp");
    //             builder.build_unreachable();
    //         }
            BrilligOpcode::Stop { return_data_offset, return_data_size } => {
                // builder.build_return(Some(&i32_type.const_int(0, false)));
                builder.build_return(None);
            }
            BrilligOpcode::Return => {
                builder.build_return(None);
            }
            _ => unimplemented!("Unimplemented enum variant: {:?}", opcode),
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

    print!("{}", module.print_to_string().to_string());

}