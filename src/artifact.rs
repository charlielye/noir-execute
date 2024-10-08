use acvm::acir::circuit::brillig::BrilligFunctionId;
use acvm::FieldElement;
use log::debug;
use acvm::acir::brillig::Opcode as BrilligOpcode;
use acvm::acir::circuit::{Opcode, Program};
use std::io::Read;
use serde::{Deserialize, Serialize};
use noirc_errors::debug_info::ProgramDebugInfo;

#[derive(Debug, Serialize, Deserialize)]
pub struct CompiledArtifact {
    pub noir_version: String,
    // pub abi: serde_json::Value,
    pub abi: noirc_abi::Abi,
    #[serde(
        serialize_with = "Program::serialize_program_base64",
        deserialize_with = "Program::deserialize_program_base64"
    )]
    pub bytecode: Program<FieldElement>,
    #[serde(
        serialize_with = "ProgramDebugInfo::serialize_compressed_base64_json",
        deserialize_with = "ProgramDebugInfo::deserialize_compressed_base64_json"
    )]
    pub debug_symbols: ProgramDebugInfo,
    pub file_map: serde_json::Value,
}

/// Extract the Brillig program from its `Program` wrapper.
/// Noir entry point unconstrained functions are compiled to their own list contained
/// as part of a full program. Function calls are then accessed through a function
/// pointer opcode in ACIR that fetches those unconstrained functions from the main list.
/// This function just extracts Brillig bytecode, with the assumption that the
/// 0th unconstrained function in the full `Program` structure.
pub fn extract_brillig_from_acir_program(
    program: &Program<FieldElement>,
) -> &[BrilligOpcode<FieldElement>] {
    assert_eq!(
        program.functions.len(),
        1,
        "An AVM program should have only a single ACIR function with a 'BrilligCall'"
    );
    let main_function = &program.functions[0];
    let opcodes = &main_function.opcodes;
    assert_eq!(opcodes.len(), 1, "An AVM program should only have a single `BrilligCall`");
    match opcodes[0] {
        Opcode::BrilligCall { id, .. } => assert_eq!(id, BrilligFunctionId(0), "The ID of the `BrilligCall` must be 0 as we have a single `Brillig` function"),
        _ => panic!("Tried to extract a Brillig program from its ACIR wrapper opcode, but the opcode doesn't contain Brillig!"),
    }
    assert_eq!(
        program.unconstrained_functions.len(),
        1,
        "An AVM program should be contained entirely in only a single `Brillig` function"
    );
    &program.unconstrained_functions[0].bytecode
}

/// Print inputs, outputs, and instructions in a Brillig program
pub fn dbg_print_brillig_program(brillig_bytecode: &[BrilligOpcode<FieldElement>]) {
    debug!("Printing Brillig program...");
    for (i, instruction) in brillig_bytecode.iter().enumerate() {
        debug!("\tPC:{0} {1:?}", i, instruction);
    }
}