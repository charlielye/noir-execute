#![allow(warnings)]
#![warn(clippy::semicolon_if_nothing_returned)]
#![cfg_attr(not(test), warn(unused_crate_dependencies, unused_extern_crates))]

use acvm::FieldElement;
use base64::write;
use brillig::BinaryFieldOp;
use brillig::BinaryIntOp;
use brillig::BitSize;
use brillig::HeapArray;
use brillig::IntegerBitSize;
use brillig::MemoryAddress;
use brillig::Opcode;
use brillig::ValueOrArray;
use env_logger::Env;
use inkwell::intrinsics::Intrinsic;
use inkwell::targets::CodeModel;
use inkwell::targets::FileType;
use inkwell::targets::InitializationConfig;
use inkwell::targets::RelocMode;
use inkwell::targets::Target;
use inkwell::targets::TargetMachine;
use inkwell::targets::TargetTriple;
use inkwell::types::BasicMetadataTypeEnum;
use inkwell::values::AnyValue;
use inkwell::values::BasicMetadataValueEnum;
use inkwell::values::BasicValue;
use inkwell::values::PointerValue;
use inkwell::OptimizationLevel;
use log::warn;
use noirc_abi::Sign;
use std::cmp;
use std::collections::HashMap;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fmt::Pointer;
use std::fs;
use std::hash::Hash;
use std::mem;
use std::ops::Deref;
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

use crate::run::RunArgs;

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
                        if (!basic_blocks.contains_key(jump_loc) && !to_process.contains(jump_loc)) {
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
                        if (!basic_blocks.contains_key(jump_loc) && !to_process.contains(jump_loc)) {
                            to_process.push(*jump_loc);
                        }
                        let after_loc = start_location + index + 1;
                        if (!basic_blocks.contains_key(&after_loc) && !to_process.contains(&after_loc)) {
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

pub fn generate_llvm_ir(opcodes: &Vec<BrilligOpcode>, calldata_fields: &Vec<FieldElement>, run_args: &RunArgs) {
    let prelude_start = Instant::now();
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
    let bn254_fr_normalize = module.add_function("bn254_fr_normalize", context.void_type().fn_type(&[i64_ptr_type.into()], false), None);
    let bn254_fr_add = module.add_function("bn254_fr_add", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_sub = module.add_function("bn254_fr_sub", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_mul = module.add_function("bn254_fr_mul", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_div = module.add_function("bn254_fr_div", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_eq = module.add_function("bn254_fr_eq", i1_type.fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_leq = module.add_function("bn254_fr_leq", i1_type.fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let bn254_fr_lt = module.add_function("bn254_fr_lt", i1_type.fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let printf_func = module.add_function("printf", i32_type.fn_type(&[i8_ptr_type.into()], true), None);
    let print_fields_func = module.add_function("print_u256", context.void_type().fn_type(&[i64_ptr_type.into(), i64_type.into()], false), None);
    let to_radix_func = module.add_function("to_radix", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_type.into(), i64_type.into()], false), None);
    let sha256_func = module.add_function("blackbox_sha256", context.void_type().fn_type(&[i8_ptr_type.into(), i64_type.into(), i8_ptr_type.into()], false), None);
    let keccak1600_func = module.add_function("blackbox_keccak1600", context.void_type().fn_type(&[i8_ptr_type.into(), i64_type.into(), i8_ptr_type.into()], false), None);
    let blake2s_func = module.add_function("blackbox_blake2s", context.void_type().fn_type(&[i8_ptr_type.into(), i64_type.into(), i8_ptr_type.into()], false), None);
    let blake3_func = module.add_function("blackbox_blake3", context.void_type().fn_type(&[i8_ptr_type.into(), i64_type.into(), i8_ptr_type.into()], false), None);
    let pedersen_hash_func = module.add_function("blackbox_pedersen_hash", context.void_type().fn_type(&[i8_ptr_type.into(), i64_type.into(), i64_type.into(), i8_ptr_type.into()], false), None);
    let pedersen_commit_func = module.add_function("blackbox_pedersen_commit", context.void_type().fn_type(&[i8_ptr_type.into(), i64_type.into(), i64_type.into(), i8_ptr_type.into()], false), None);
    let aes_encrypt_func = module.add_function("blackbox_aes_encrypt", context.void_type().fn_type(&[i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into(), i64_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let secp256k1_func = module.add_function("blackbox_secp256k1_verify_signature", context.void_type().fn_type(&[i64_ptr_type.into(), i64_type.into(), i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let secp256r1_func = module.add_function("blackbox_secp256r1_verify_signature", context.void_type().fn_type(&[i64_ptr_type.into(), i64_type.into(), i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into(), i64_ptr_type.into()], false), None);
    let exit_fn = module.add_function("exit", context.void_type().fn_type(&[i32_type.into()], false), None);

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

    // Define memory (as 256 bit slots).
    let memory_size = 2048*2*2*2*2; // 32k words, 1mb.
    let memory_type = v256_type.array_type(memory_size);
    let memory_global = module.add_global(memory_type, Some(AddressSpace::default()), "memory");
    memory_global.set_alignment(32);
    memory_global.set_initializer(&v256_type.const_array(&vec![v256_type.const_zero(); memory_size as usize]));
    let memory = memory_global.as_pointer_value();

    // Define heap (as 256 bit slots).
    // The heap isn't directly referenced, it's referenced via &memory.
    // This is just reserving the space.
    let heap_size = 1024*1024; //*256;//*8; // 64GB for testing blob-lib.
    let heap_type = v256_type.array_type(heap_size);
    let heap_global = module.add_global(heap_type, Some(AddressSpace::default()), "heap");
    heap_global.set_alignment(32);
    // TODO: We shouldn't need to do this surely.
    heap_global.set_initializer(&v256_type.const_array(&vec![v256_type.const_zero(); heap_size as usize]));

    // Trap error string.
    let trap_str = builder.build_global_string_ptr("Trap triggered!\n", "str").unwrap().as_pointer_value();

    eprintln!("Prelude took: {:?}", prelude_start.elapsed());

    macro_rules! get_calldata_ptr_at_index {
        ($index:expr) => {
            unsafe {
                builder.build_gep(i256_type, calldata.as_pointer_value(), &[i64_type.const_int($index as u64, false)], "elem_ptr")
            }.unwrap()
        };
    }

    macro_rules! get_memory_ptr_at_index {
        ($index:expr) => {
            unsafe {
                builder.build_gep(i256_type, memory, &[i64_type.const_int($index as u64, false)], "elem_ptr")
            }.unwrap()
        };
    }

    macro_rules! get_memory_ptr_at_index_obj {
        ($index:expr) => {
            unsafe {
                builder.build_gep(i256_type, memory, &[$index], "elem_ptr")
            }.unwrap()
        };
    }

    macro_rules! get_deref_memory_ptr_at_index {
        ($index:expr) => {{
            let ptr_ptr = get_memory_ptr_at_index!($index);
            let ptr = builder.build_load(i256_type, ptr_ptr, "deref_ptr").unwrap();
            get_memory_ptr_at_index_obj!(ptr.into_int_value())
        }}
    }

    let dis_start = Instant::now();
    let opcode_map = disassemble_brillig(opcodes);
    eprintln!("Dissassembly took: {:?}", dis_start.elapsed());

    let build_start = Instant::now();

    // Pre-add all functions to the module, so we can reference them in calls.
    for (function_location, blocks) in &opcode_map {
        let function_name = format!("function_at_{}", function_location);
        let function_type = context.void_type().fn_type(&[], false);
        module.add_function(&function_name, function_type, None);
    }

    eprint!("Transpiling: {}", if run_args.verbose { "\n" } else { "" });
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
                // let before_count = block.get_instructions().count();

                match opcode {
                    BrilligOpcode::Const { destination, bit_size, value } => {
                        let limbs = field_element_to_u64_limbs(*value);
                        let const_val = VectorType::const_vector(&limbs.iter().map(|&x| i64_type.const_int(x as u64, false)).collect::<Vec<_>>());

                        let dest_index = destination.0;
                        let dest_ptr = get_memory_ptr_at_index!(dest_index);
                        builder.build_store(dest_ptr, const_val).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::CalldataCopy { destination_address, size, offset } => {
                        for i in 0..*size {
                            let addr = destination_address.0 + i;
                            let src_ptr = get_calldata_ptr_at_index!(i + offset);
                            let dest_ptr = get_memory_ptr_at_index!(addr);
                            let value = builder.build_load(v256_type, src_ptr, "cdc_val").unwrap();
                            builder.build_store(dest_ptr, value).unwrap().set_alignment(32);
                        }
                    }
                    BrilligOpcode::ConditionalMov { destination, source_a, source_b, condition } => {
                        let cond_index = condition.0;
                        let src_a_index = source_a.0;
                        let src_b_index = source_b.0;
                        let dest_index = destination.0;

                        let cond_ptr = get_memory_ptr_at_index!(cond_index);
                        let src_a_ptr = get_memory_ptr_at_index!(src_a_index);
                        let src_b_ptr = get_memory_ptr_at_index!(src_b_index);
                        let dest_ptr = get_memory_ptr_at_index!(dest_index);

                        let cond_val = builder.build_load(i256_type, cond_ptr, "cond_val").unwrap().into_int_value();
                        let src_a_val = builder.build_load(v256_type, src_a_ptr, "src_a_val").unwrap();
                        let src_b_val = builder.build_load(v256_type, src_b_ptr, "src_b_val").unwrap();

                        let zero = i256_type.const_int(0, false);
                        let cmp = builder.build_int_compare(IntPredicate::NE, cond_val, zero, "cmp").unwrap();

                        let selected_val = builder.build_select(cmp, src_a_val, src_b_val, "selected_val").unwrap();
                        builder.build_store(dest_ptr, selected_val).unwrap().set_alignment(32);
                    },
                    BrilligOpcode::Mov { destination, source } => {
                        let src_index = source.0;
                        let dest_index = destination.0;
                        let src_ptr = get_memory_ptr_at_index!(src_index);
                        let dest_ptr = get_memory_ptr_at_index!(dest_index);
                        let value = builder.build_load(v256_type, src_ptr, "mov_val").unwrap();
                        builder.build_store(dest_ptr, value).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::Cast { destination, source, bit_size } => {
                        let src_index = source.0;
                        let dest_index = destination.0;
                        let src_ptr = get_memory_ptr_at_index!(src_index);
                        let dest_ptr = get_memory_ptr_at_index!(dest_index);

                        let itype = match bit_size {
                            BitSize::Integer(IntegerBitSize::U1) => i8_type,
                            BitSize::Integer(IntegerBitSize::U8) => i8_type,
                            BitSize::Integer(IntegerBitSize::U16) => i16_type,
                            BitSize::Integer(IntegerBitSize::U32) => i32_type,
                            BitSize::Integer(IntegerBitSize::U64) => i64_type,
                            BitSize::Integer(IntegerBitSize::U128) => i128_type,
                            BitSize::Field => i256_type,
                            _ => unreachable!("Unsupported bit size: {:?}", bit_size)
                        };

                        // If casting to an int, ensure we're not in montgomery form.
                        // Bit 255 is set in fields in montgomery form.
                        if let BitSize::Integer(..) = bit_size {
                            builder.build_call(bn254_fr_normalize, &[src_ptr.into()], "fr_norm_call");
                        }

                        let value = builder.build_load(itype, src_ptr, "cast_val").unwrap();
                        // Zero destination.
                        builder.build_store(dest_ptr, v256_type.const_zero()).unwrap().set_alignment(32);

                        if let BitSize::Integer(IntegerBitSize::U1) = bit_size {
                            // For 1-bit size, mask the value to keep only the LSB.
                            let masked_value = builder.build_and(value.into_int_value(), i8_type.const_int(1, false), "masked").unwrap();
                            builder.build_store(dest_ptr, masked_value).unwrap().set_alignment(32);
                        } else {
                            builder.build_store(dest_ptr, value).unwrap().set_alignment(32);
                        }
                    }
                    BrilligOpcode::Store { destination_pointer, source } => {
                        let src_index = source.0;
                        let src_value_ptr = get_memory_ptr_at_index!(src_index);
                        let src_value = builder.build_load(v256_type, src_value_ptr, "store_val").unwrap();

                        let dest_ptr = get_deref_memory_ptr_at_index!(destination_pointer.0);
                        builder.build_store(dest_ptr, src_value).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::Load { destination, source_pointer } => {
                        let dest_index = destination.0;
                        let dest_ptr = get_memory_ptr_at_index!(dest_index);

                        let src_ptr_index = source_pointer.0;
                        let src_ptr = get_deref_memory_ptr_at_index!(src_ptr_index);
                        let src_value = builder.build_load(v256_type, src_ptr, "load_val").unwrap();

                        builder.build_store(dest_ptr, src_value).unwrap().set_alignment(32);
                    }
                    BrilligOpcode::BinaryFieldOp { destination, op, lhs, rhs } => {
                        let lhs_index = lhs.0;
                        let rhs_index = rhs.0;
                        let result_index = destination.0;

                        let lhs_ptr = get_memory_ptr_at_index!(lhs_index);
                        let rhs_ptr = get_memory_ptr_at_index!(rhs_index);
                        let result_ptr = get_memory_ptr_at_index!(result_index);

                        // mov_is_int!(lhs_index, result_index);

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
                                builder.build_call(bn254_fr_eq, &[lhs_ptr.into(), rhs_ptr.into(), result_ptr.into()], "eq_call");
                            }
                            BinaryFieldOp::LessThan => {
                                builder.build_call(bn254_fr_lt, &[lhs_ptr.into(), rhs_ptr.into(), result_ptr.into()], "lt_call");
                            }
                            BinaryFieldOp::LessThanEquals => {
                                builder.build_call(bn254_fr_leq, &[lhs_ptr.into(), rhs_ptr.into(), result_ptr.into()], "leq_call");
                            }
                            _ => unimplemented!("Unimplemented BinaryFieldOp variant: {:?}", op),
                        }
                    }
                    BrilligOpcode::BinaryIntOp { destination, op, bit_size, lhs, rhs } => {
                        let lhs_index = lhs.0;
                        let rhs_index = rhs.0;
                        let result_index = destination.0;

                        let lhs_ptr = get_memory_ptr_at_index!(lhs_index);
                        let rhs_ptr = get_memory_ptr_at_index!(rhs_index);
                        let result_ptr = get_memory_ptr_at_index!(result_index);

                        let itype = match bit_size {
                            IntegerBitSize::U1 => i1_type,
                            IntegerBitSize::U8 => i8_type,
                            IntegerBitSize::U16 => i16_type,
                            IntegerBitSize::U32 => i32_type,
                            IntegerBitSize::U64 => i64_type,
                            IntegerBitSize::U128 => i128_type,
                            _ => unreachable!("Unsupported bit size: {bit_size}")
                        };
                        // let itype = i256_type;

                        let bit_size = match bit_size {
                            IntegerBitSize::U1 => 1,
                            IntegerBitSize::U8 => 8,
                            IntegerBitSize::U16 => 16,
                            IntegerBitSize::U32 => 32,
                            IntegerBitSize::U64 => 64,
                            IntegerBitSize::U128 => 128,
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
                                // builder.build_left_shift(lhs_val, rhs_val, "bio_shl").unwrap()
                                let bs = itype.const_int(bit_size, false);
                                let cmp = builder.build_int_compare(IntPredicate::UGE, rhs_val, bs, "cmp_ge_bs").unwrap();
                                let zero = itype.const_int(0, false);
                                let shifted = builder.build_left_shift(lhs_val, rhs_val, "bio_shl").unwrap();
                                builder.build_select(cmp, zero, shifted, "select_shl").unwrap().into_int_value()
                            }
                            BinaryIntOp::Shr => {
                                // builder.build_right_shift(lhs_val, rhs_val, false, "bio_shr").unwrap()
                                let bs = itype.const_int(bit_size, false);
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

                        // Zero destination for < 256 bits?
                        // TODO: maybe can make assumptions
                        builder.build_store(result_ptr, v256_type.const_zero()).unwrap().set_alignment(32);

                        builder.build_store(result_ptr, value).unwrap().set_alignment(32);
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
                        let cond_ptr = get_memory_ptr_at_index!(cond_index);
                        let cond_val = builder.build_load(i1_type, cond_ptr, "ji_cond").unwrap().into_int_value();
                        builder.build_conditional_branch(cond_val, block_map[location], block_map[&(block_location+opcode_index+1)]);
                    }
                    BrilligOpcode::JumpIfNot { condition, location } => {
                        let cond_index = condition.0;
                        let cond_ptr = get_memory_ptr_at_index!(cond_index);
                        let cond_val = builder.build_load(i1_type, cond_ptr, "jin_cond").unwrap().into_int_value();
                        builder.build_conditional_branch(cond_val.const_not(), block_map[location], block_map[&(block_location+opcode_index+1)]);
                    }
                    BrilligOpcode::Trap { revert_data } => {
                        let ptr = get_memory_ptr_at_index!(revert_data.pointer.0);
                        let size = i64_type.const_int(revert_data.size as u64, false);
                        builder.build_call(print_fields_func, &[ ptr.into(), size.into()], "print_fields_call");

                        if (run_args.ill_trap) {
                            let trap_intrinsic = Intrinsic::find("llvm.trap").unwrap();
                            let trap_function = trap_intrinsic.get_declaration(&module, &[]).unwrap();
                            builder.build_call(trap_function, &[], "trap_call");
                        } else {
                            builder.build_call(printf_func, &[trap_str.into()], "printf_call");
                            builder.build_call(exit_fn, &[i32_type.const_int(1, false).into()], "exit_call");
                        }

                        builder.build_unreachable();
                    }
                    BrilligOpcode::Stop { return_data_offset, return_data_size } => {
                        // let ptr = get_memory_at_index!(*return_data_offset);
                        // let size = i64_type.const_int(*return_data_size as u64, false);
                        // builder.build_call(print_fields_func, &[ ptr.into(), size.into()], "print_fields_call");
                        builder.build_return(None);
                    }
                    BrilligOpcode::Return => {
                        builder.build_return(None);
                    }
                    BrilligOpcode::BlackBox(op) => {
                        match op {
                            brillig::BlackBoxOp::ToRadix { input, radix, output } => {
                                let input_ptr = get_memory_ptr_at_index!(input.0);
                                let output_ptr_ptr = get_memory_ptr_at_index!(output.pointer.0);
                                let output_ptr = builder.build_load(i256_type, output_ptr_ptr, "bb_tr_out_ptr").unwrap();
                                let output_gep = get_memory_ptr_at_index_obj!(output_ptr.into_int_value());
                                let size = i64_type.const_int(output.size as u64, false);
                                let radix = i64_type.const_int(*radix as u64, false);
                                builder.build_call(
                                    to_radix_func,
                                    &[ input_ptr.into(), output_gep.into(), size.into(), radix.into() ],
                                    "to_radix_call",
                                );
                            }
                            brillig::BlackBoxOp::Blake2s { message, output } => {
                                let message_ptr = get_deref_memory_ptr_at_index!(message.pointer.0);
                                let message_size_ptr = get_memory_ptr_at_index!(message.size.0);
                                let message_size = builder.build_load(i64_type, message_size_ptr, "bb_blake2s_msg_size").unwrap();
                                let output_ptr = get_deref_memory_ptr_at_index!(output.pointer.0);

                                builder.build_call(
                                    blake2s_func,
                                    &[ message_ptr.into(), message_size.into(), output_ptr.into() ],
                                    "blake2s_call",
                                );
                            }
                            brillig::BlackBoxOp::Blake3 { message, output } => {
                                let message_ptr = get_deref_memory_ptr_at_index!(message.pointer.0);
                                let message_size_ptr = get_memory_ptr_at_index!(message.size.0);
                                let message_size = builder.build_load(i64_type, message_size_ptr, "bb_blake2s_msg_size").unwrap();
                                let output_ptr = get_deref_memory_ptr_at_index!(output.pointer.0);

                                builder.build_call(
                                    blake3_func,
                                    &[ message_ptr.into(), message_size.into(), output_ptr.into() ],
                                    "blake3_call",
                                );
                            }
                            brillig::BlackBoxOp::Keccakf1600 { message, output } => {
                                let message_ptr = get_deref_memory_ptr_at_index!(message.pointer.0);
                                let message_size_ptr = get_memory_ptr_at_index!(message.size.0);
                                let message_size = builder.build_load(i64_type, message_size_ptr, "bb_keccak_msg_size").unwrap();
                                let output_ptr = get_deref_memory_ptr_at_index!(output.pointer.0);

                                builder.build_call(
                                    keccak1600_func,
                                    &[ message_ptr.into(), message_size.into(), output_ptr.into() ],
                                    "keccak1600_call",
                                );
                            },
                            brillig::BlackBoxOp::Sha256 { message, output } => {
                                let message_ptr_ptr = get_memory_ptr_at_index!(message.pointer.0);
                                let message_ptr = builder.build_load(i256_type, message_ptr_ptr, "bb_sha_msg_ptr").unwrap();
                                let message_gep = get_memory_ptr_at_index_obj!(message_ptr.into_int_value());

                                let message_size_ptr = get_memory_ptr_at_index!(message.size.0);
                                let message_size = builder.build_load(i64_type, message_size_ptr, "bb_sha_msg_size").unwrap();

                                let output_ptr_ptr = get_memory_ptr_at_index!(output.pointer.0);
                                let output_ptr = builder.build_load(i256_type, output_ptr_ptr, "bb_sha_out_ptr").unwrap();
                                let output_gep = get_memory_ptr_at_index_obj!(output_ptr.into_int_value());

                                builder.build_call(
                                    sha256_func,
                                    &[ message_gep.into(), message_size.into(), output_gep.into() ],
                                    "sha256_call",
                                );
                            },
                            brillig::BlackBoxOp::PedersenHash { inputs, domain_separator, output } => {
                                let inputs_ptr_ptr = get_memory_ptr_at_index!(inputs.pointer.0);
                                let inputs_ptr = builder.build_load(i256_type, inputs_ptr_ptr, "bb_ped_hash_in_ptr").unwrap();
                                let inputs_gep = get_memory_ptr_at_index_obj!(inputs_ptr.into_int_value());

                                let inputs_size_ptr = get_memory_ptr_at_index!(inputs.size.0);
                                let inputs_size = builder.build_load(i64_type, inputs_size_ptr, "bb_ped_hash_in_size").unwrap();

                                let separator = builder.build_load(i64_type, get_memory_ptr_at_index!(domain_separator.0), "bb_ped_hash_in_size").unwrap();

                                let output_ptr = get_memory_ptr_at_index!(output.0);

                                builder.build_call(
                                    pedersen_hash_func,
                                    &[ inputs_gep.into(), inputs_size.into(), separator.into(), output_ptr.into() ],
                                    "pedersen_hash_call",
                                );
                            },
                            brillig::BlackBoxOp::PedersenCommitment { inputs, domain_separator, output } => {
                                let inputs_ptr_ptr = get_memory_ptr_at_index!(inputs.pointer.0);
                                let inputs_ptr = builder.build_load(i256_type, inputs_ptr_ptr, "bb_ped_commit_in_ptr").unwrap();
                                let inputs_gep = get_memory_ptr_at_index_obj!(inputs_ptr.into_int_value());

                                let inputs_size_ptr = get_memory_ptr_at_index!(inputs.size.0);
                                let inputs_size = builder.build_load(i64_type, inputs_size_ptr, "bb_ped_commit_in_size").unwrap();

                                let separator = builder.build_load(i64_type, get_memory_ptr_at_index!(domain_separator.0), "bb_ped_commit_in_size").unwrap();

                                let output_ptr_ptr = get_memory_ptr_at_index!(output.pointer.0);
                                let output_ptr = builder.build_load(i256_type, output_ptr_ptr, "bb_ped_commit_in_ptr").unwrap();
                                let output_gep = get_memory_ptr_at_index_obj!(output_ptr.into_int_value());

                                builder.build_call(
                                    pedersen_commit_func,
                                    &[ inputs_gep.into(), inputs_size.into(), separator.into(), output_gep.into() ],
                                    "pedersen_commit_call",
                                );
                            },
                            brillig::BlackBoxOp::AES128Encrypt { inputs, iv, key, outputs } => {
                                let inputs_ptr = get_deref_memory_ptr_at_index!(inputs.pointer.0);
                                let inputs_size_ptr = get_memory_ptr_at_index!(inputs.size.0);
                                let inputs_size = builder.build_load(i64_type, inputs_size_ptr, "bb_aes_encrypt_in_size").unwrap();

                                let iv_ptr = get_deref_memory_ptr_at_index!(iv.pointer.0);

                                let key_ptr_ptr = get_memory_ptr_at_index!(key.pointer.0);
                                let key_ptr = builder.build_load(i256_type, key_ptr_ptr, "bb_aes_encrypt_key_ptr").unwrap();
                                let key_gep = get_memory_ptr_at_index_obj!(key_ptr.into_int_value());

                                let output_ptr_ptr = get_memory_ptr_at_index!(outputs.pointer.0);
                                let output_ptr = builder.build_load(i256_type, output_ptr_ptr, "bb_aes_encrypt_out_ptr").unwrap();
                                let output_gep = get_memory_ptr_at_index_obj!(output_ptr.into_int_value());

                                let output_size_ptr = get_memory_ptr_at_index!(outputs.size.0);

                                builder.build_call(
                                    aes_encrypt_func,
                                    &[ inputs_ptr.into(), iv_ptr.into(), key_gep.into(), inputs_size.into(), output_gep.into(), output_size_ptr.into() ],
                                    "aes_encrypt_call",
                                );
                            },
                            brillig::BlackBoxOp::EcdsaSecp256r1 { hashed_msg, public_key_x, public_key_y, signature, result } |
                            brillig::BlackBoxOp::EcdsaSecp256k1 { hashed_msg, public_key_x, public_key_y, signature, result } => {
                                let message_ptr = get_deref_memory_ptr_at_index!(hashed_msg.pointer.0);
                                let message_size_ptr = get_memory_ptr_at_index!(hashed_msg.size.0);
                                let message_size = builder.build_load(i64_type, message_size_ptr, "bb_sha_msg_size").unwrap();
                                let x_ptr = get_deref_memory_ptr_at_index!(public_key_x.pointer.0);
                                let y_ptr = get_deref_memory_ptr_at_index!(public_key_y.pointer.0);
                                let sig_ptr = get_deref_memory_ptr_at_index!(signature.pointer.0);
                                let result_ptr = get_memory_ptr_at_index!(result.0);

                                builder.build_call(
                                    if matches!(op, brillig::BlackBoxOp::EcdsaSecp256k1 { .. }) { secp256k1_func } else { secp256r1_func },
                                    &[ message_ptr.into(), message_size.into(), x_ptr.into(), y_ptr.into(), sig_ptr.into(), result_ptr.into() ],
                                    "secp256_call",
                                );
                            },
                            _ => unimplemented!("Unimplemented BlackBox: {:?}", op)
                        }
                    }
                    BrilligOpcode::ForeignCall { function, destinations, destination_value_types, inputs, input_value_types} => {
                        match function.as_str() {
                            "print" => {
                                // let value = if let ValueOrArray::HeapArray(v) = inputs[1] { v } else { unreachable!() };
                                match inputs[1] {
                                    ValueOrArray::HeapArray(v) =>  {
                                        let lhs_ptr = get_memory_ptr_at_index!(v.pointer.0);
                                        let ptr_val = builder.build_load(i256_type, lhs_ptr, "fc_val_ptr").unwrap();
                                        let ptr_value = get_memory_ptr_at_index_obj!(ptr_val.into_int_value());
                                        builder.build_call(print_fields_func, &[ ptr_value.into(), i64_type.const_int(v.size as u64, false).into()], "print_fields_call");
                                    }
                                    ValueOrArray::MemoryAddress(v) => {
                                        let ptr = get_memory_ptr_at_index!(v.0);
                                        builder.build_call(print_fields_func, &[ ptr.into(), i64_type.const_int(1, false).into()], "print_fields_call");
                                    }
                                    _ => unimplemented!("Unknown: {:?}", inputs[1]),
                                };
                            }
                            _ => {
                                let mut function = "foreign_call_".to_owned() + function;
                                let mut args: Vec<BasicMetadataValueEnum> = Vec::new();
                                let mut types: Vec<BasicMetadataTypeEnum> = Vec::new();

                                let mut inputs = inputs.clone();
                                inputs.pop();
                                inputs.pop();
                                // function += if inputs.len() > 0 { "_" } else { "" };

                                for input in inputs.iter().chain(destinations.iter()) {
                                    match input {
                                        ValueOrArray::MemoryAddress(address) => {
                                            args.push(get_memory_ptr_at_index!(address.0).into());
                                            types.push(i64_ptr_type.into());
                                            // function += "s";
                                        }
                                        ValueOrArray::HeapArray(array) => {
                                            let ptr = builder.build_load(i64_type, get_memory_ptr_at_index!(array.pointer.0), "fc_arr_ptr").unwrap();
                                            args.push(get_memory_ptr_at_index_obj!(ptr.into_int_value()).into());
                                            args.push(i64_type.const_int(array.size as u64, false).into());
                                            types.push(i64_ptr_type.into());
                                            types.push(i64_type.into());
                                            // function += "a";
                                        },
                                        ValueOrArray::HeapVector(vector) => {
                                            args.push(get_memory_ptr_at_index!(vector.pointer.0).into());
                                            let size_ptr = get_memory_ptr_at_index!(vector.size.0);
                                            let size = builder.build_load(i64_type, size_ptr, "fc_heap_vec_size").unwrap();
                                            args.push(size.into());
                                            types.push(i64_ptr_type.into());
                                            types.push(i64_type.into());
                                            // function += "a";
                                        },
                                    }
                                }

                                let fn_type = context.void_type().fn_type(&types, false);
                                let foreign_func = module.get_function(&function).unwrap_or_else(|| {
                                    module.add_function(&function, fn_type, None) });
                                builder.build_call(foreign_func, &args, "foreign_call_result");

                            }
                            // _ => eprintln!("Skipping ForeignCall: {:?}", function),
                        }
                    }
                    _ => unimplemented!("Unimplemented enum variant: {:?}", opcode),
                }

                // let inst_count = block.get_instructions().count() - before_count;
                // let opcode_str = &format!("{:?}", opcode);
                // eprintln!("{:?} {:0>3} ({:0>2}.{:0>2}.{:0>2}): {opcode_str}", op_start.elapsed(), block_location + opcode_index, function_location, block_location, opcode_index);
                if (run_args.verbose) {
                    let opcode_str = &format!("{:?}", opcode);
                    eprintln!("{:0>3} ({:0>2}.{:0>2}.{:0>2}): {opcode_str}", block_location + opcode_index, function_location, block_location, opcode_index);
                } else if ((block_location + opcode_index) % 100_000 == 0) {
                    eprint!(".");
                }
            }
        }
    }
    eprintln!("");

    builder.position_at_end(entry_block);
    let function = module.get_function("function_at_0").unwrap();
    builder.build_call(function, &[], "calltmp");

    builder.build_return(Some(&i32_type.const_int(0, false)));

    eprintln!("Transpile took: {:?}", build_start.elapsed());

    module.verify().unwrap();

    let write_start = Instant::now();
    if (run_args.write_ll) {
        let ll = module.print_to_string().to_string();
        std::fs::write("program.ll", ll).expect("Failed to write ll file.");
        eprintln!("Write took: {:?}", write_start.elapsed());
    } else {
        // Compile directly to native binary without validation and writing to .ll file.
        eprintln!("Compiling...");
        let target_triple = TargetTriple::create("x86_64-pc-linux-gnu");
        Target::initialize_native(&InitializationConfig::default()).expect("Failed to initialize native target");
        let target = Target::from_triple(&target_triple).unwrap();
        let target_machine = target.create_target_machine(&target_triple, "generic", "", OptimizationLevel::None, RelocMode::PIC, CodeModel::Large).unwrap();
        let obj_file = target_machine.write_to_memory_buffer(&module, FileType::Object).unwrap();
        std::fs::write("program.o", obj_file.as_slice()).expect("Failed to write object file.");
        eprintln!("Compilation took: {:?}", write_start.elapsed());
    }

    // SLOWER :/ (even without +avx)
    //---------------------------------------------------------------------------------
    // // Initialize all targets for LLVM
    // Target::initialize_all(&InitializationConfig::default());

    // // Retrieve the default target triple for the current host
    // let target_triple = TargetMachine::get_default_triple();
    // let target = Target::from_triple(&target_triple).expect("Error retrieving target from triple");

    // // Retrieve the best CPU for the current host
    // let host_cpu = TargetMachine::get_host_cpu_name().to_string();

    // // Create a target machine for the current host with optimal settings
    // let target_machine = target.create_target_machine(
    //     &target_triple,
    //     &host_cpu,
    //     "+avx",
    //     OptimizationLevel::None,
    //     RelocMode::PIC,
    //     CodeModel::Default
    // ).expect("Failed to create target machine");
}
