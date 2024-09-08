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
use clap::Parser;
use env_logger::Env;
use inkwell::intrinsics::Intrinsic;
use inkwell::values::AnyValue;
use inkwell::values::PointerValue;
use log::warn;
use run::RunArgs;
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
use std::process;
use ark_ff::BigInt;

use inkwell::context::Context;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::module::Module;
use inkwell::types::{BasicTypeEnum, IntType, VectorType};
use inkwell::values::{BasicValueEnum, IntValue};
use inkwell::AddressSpace;
use inkwell::IntPredicate;

mod run;
mod artifact;
mod codegen;

// use artifact::CompiledArtifact;
use run::run;

type BrilligOpcode = brillig::Opcode<FieldElement>;

fn main() {
    let args: Vec<String> = env::args().collect();
    let project_path = Path::new(&args[1]);
    let run_args = RunArgs::parse();
    let result = run(run_args);

    match result {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error: {:?}", e);
            process::exit(1);
        },
    }
}