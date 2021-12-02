virtual_modular is an experimental software modular synth and livecoding environment intended to be used for generative composition, sound design and dynamic music in interactive projects.

In its current form it's exploratory and while some parts are relatively polished, others are largely dumping grounds for ideas. It is finished enough to be used for composition and sound design, both interactively and as an embedded sound engine. [This playlist](https://soundcloud.com/user-166463215/sets/realtime-synth) contains some example tracks which are raw recordings of the synth with no post-processing.

# To Use

The most common entry point to the system will be the [virtual_modular_dynamic_environment](./dynamic_environment) executable which provides interactive playback of patches with dynamic reloading. It can be used for sound design or composition. There are several examples of patches in that crate's [examples directory](./dynamic_environment/examples).

For use as an embedded sound engine it is possible but difficult to write patches by hand using the rust API. It is easier to design the patches using the dynamic environment and then compile them to rust using the code generator in the [virtual_modular_definition_language](./definition_language) crate.
