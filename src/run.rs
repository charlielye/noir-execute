use std::{collections::BTreeMap, io::Error, path::{Path, PathBuf}, time::Instant};
use acvm::{acir::native_types::WitnessStack, FieldElement};
use clap::{Args, Parser};
use nargo::package::Package;
use nargo_toml::{get_package_manifest, resolve_workspace_from_toml, ManifestError, PackageSelection};
use noirc_abi::{errors::{AbiError, InputParserError}, input_parser::{Format, InputValue}, Abi, InputMap};
use thiserror::Error;

use crate::{artifact::CompiledArtifact, codegen::generate_llvm_ir};

/// Executes a circuit to calculate its return value
#[derive(Parser, Debug, Clone)]
pub(crate) struct RunArgs {
    #[arg(long, short)]
    project_dir: PathBuf,

    /// The name of the toml file which contains the inputs for the prover
    #[arg(long, default_value = "Prover")]
    prover_name: String,

    /// The name of the package to execute
    #[arg(long, conflicts_with = "workspace")]
    package: Option<String>,

    /// Execute all packages in the workspace
    #[arg(long, conflicts_with = "package")]
    workspace: bool,

    #[arg(long)]
    ill_trap: bool,
}

#[derive(Debug, Error)]
pub(crate) enum RunError {
    #[error("{0}")]
    Generic(String),

    #[error("Error: {} is not a valid path", .0.display())]
    PathNotValid(PathBuf),

    #[error("Error: could not deserialize build program: {0}")]
    ProgramSerializationError(String),

    #[error(transparent)]
    ManifestError(#[from] ManifestError),

    #[error(transparent)]
    AbiError(#[from] AbiError),

    #[error(transparent)]
    InputParserError(#[from] InputParserError),

    #[error(
      " Error: cannot find {0} in expected location {1:?}.\n Please generate this file at the expected location."
    )]
    MissingTomlFile(String, PathBuf),
}


pub(crate) fn run(args: RunArgs) -> Result<(), RunError> {
    let toml_path = get_package_manifest(args.project_dir.as_path())?;
    // let default_selection =
    //     if args.workspace { PackageSelection::All } else { PackageSelection::DefaultOrAll };
    // let selection = args.package.map_or(default_selection, PackageSelection::Selected);
    let workspace = resolve_workspace_from_toml(&toml_path, PackageSelection::All, None)?;
    let target_dir = &workspace.target_directory_path();

    let binary_packages = workspace.into_iter().filter(|package| package.is_binary());
    for package in binary_packages {
        let program_artifact_path = workspace.package_build_path(package);
        let program: CompiledArtifact = read_program_from_file(program_artifact_path)?.into();

        generate_program(
            program,
            package,
            &args.prover_name,
            args.ill_trap,
            None
            // args.oracle_resolver.as_deref(),
        )?;

        // println!("[{}] Circuit witness successfully solved", package.name);
        // if let Some(return_value) = return_value {
        //     println!("[{}] Circuit output: {return_value:?}", package.name);
        // }
        // if let Some(witness_name) = &args.witness_name {
        //     let witness_path = save_witness_to_dir(witness_stack, witness_name, target_dir)?;

        //     println!("[{}] Witness saved to {}", package.name, witness_path.display());
        // }
    }
    Ok(())
}

pub(crate) fn read_program_from_file<P: AsRef<Path>>(
  circuit_path: P,
) -> Result<CompiledArtifact, RunError> {
  let file_path = circuit_path.as_ref().with_extension("json");

  let load_start = Instant::now();
  let input_string = std::fs::read(&file_path).map_err(|_| RunError::PathNotValid(file_path))?;
  let program = serde_json::from_slice(&input_string)
      .map_err(|err| RunError::ProgramSerializationError(err.to_string()))?;
  eprintln!("Artifact load took: {:?}", load_start.elapsed());

  Ok(program)
}

fn generate_program(
  program: CompiledArtifact,
  package: &Package,
  prover_name: &str,
  ill_trap: bool,
  foreign_call_resolver_url: Option<&str>,
) -> Result<(), RunError> {
  // Parse the initial witness values from Prover.toml
  let (inputs_map, _) =
      read_inputs_from_file(&package.root_dir, prover_name, Format::Toml, &program.abi)?;

  let initial_witness = program.abi.encode(&inputs_map, None)?;
  let calldata: Vec<_> = initial_witness.into_iter().map(|(_key, value)| value).collect();


  assert!(program.bytecode.unconstrained_functions.len() == 1);

  for function in &program.bytecode.unconstrained_functions {
      generate_llvm_ir(&function.bytecode, &calldata, ill_trap);
      // println!("");
      // for opcode in &function.bytecode {
      //     println!("{:?}", opcode);
      // }
  }

  Ok(())
}

pub(crate) fn read_inputs_from_file<P: AsRef<Path>>(
  path: P,
  file_name: &str,
  format: Format,
  abi: &Abi,
) -> Result<(InputMap, Option<InputValue>), RunError> {
  if abi.is_empty() {
      return Ok((BTreeMap::new(), None));
  }

  let file_path = path.as_ref().join(file_name).with_extension(format.ext());
  if !file_path.exists() {
      return Err(RunError::MissingTomlFile(file_name.to_owned(), file_path));
  }

  let input_string = std::fs::read_to_string(file_path).unwrap();
  let mut input_map = format.parse(&input_string, abi)?;
  let return_value = input_map.remove("return");

  Ok((input_map, return_value))
}