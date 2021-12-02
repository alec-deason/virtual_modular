This crate provides the parser for the patch definition language used by [virtual_modular_dynamic_environment](../dynamic_environment) and a code generator to convert those patches to raw rust.

# To Run
```sh
cargo run /path/to/patch/definition.synth /path/to/output.rs
```

The current version outputs a block of rust which require additional `use` statements and setup to be useful. At some point I will put together a more complete build system but that is very much future work.
