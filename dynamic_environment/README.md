This crate provides an interactive version of the virtual_modular system which can be used for sound design, composition, as a midi instrument or (with some limitations) for livecoding performance.

# To Run
```sh
cargo run --release /path/to/patch/definition.synth
```

The system will play the patch until it is stopped. If the patch changes on disk it will automatically be reloaded. This allows you to work on the patch while it is playing and hear changes in real time.

The patch language is declarative and describes the patch's DSP graph as a set of nodes connected by edges.

Nodes are defined using `=`:
```
my_node=Sine(220)
```

The left-hand side is the name and the right-hand side is a expression which consists of node constructors, references to other nodes and basic arithmetic. `my_node` is declared to be a node of type `Sine` with the constant value `220` connected to it's input.

An example of a more complex expression:
```
my_second_node=my_node/2 + 1
```

This snippet declares a second node which outputs the value of the first node converted from a -1..1 sine wave into a 0..1 sine wave.

In addition to the implicit edges created by expressions it is possible to explicitly create edges:
```
(target_node, 0, source_node|0)
```

This creates an edge from the 0th output of `source_node` to the 0th input of `target_node`. If there are multiple edges connected to the same input port their values are summed. The `nodename|int` syntax specifies which output should be used in the expression. Output 0 is the default and can be omitted.

Patches have one special node which always exists named `output`. It has two inputs corresponding to the left and right channels. Any signals sent to those inputs will be played out to the computer's speakers.

There is currently no documented list of node types beyond the code in [virtual_modular_core_nodes](../core_nodes) and that, unfortunately, is unlikely to change until the system stabilises a bit more. For some examples of what is possible look at the demonstration patches in the [examples](./examples) directory.

# Livecoding limitations
There are a few gaps before the system is useful for livecoding performance. Nodes preserve their state between patch reloads unless the definition of the node itself changes in any way in which case it loses all state. This is most problematic with sequencers which lose their place and will become out of sync with other sequences after they are reloaded. There is also no way to stage or beat sync changes to the graph. I plan to work on both those issues in the future because it would be super neat if this was a performable instrument.

# Performance and embeddability

This system performs well enough to play reasonably complex patches in real time on a laptop but does sacrifice some speed for the ability to dynamically reload. It is also intended only as an interactive interface for developing patches and not to be embedded in other applications to playback patches. For embedded playback you should compile the patch definition to rust which can then be used in any application and will have better performance than the dynamic environment. There is a code generator in [virtual_modular_definition_language](../definition_language) for doing that.
