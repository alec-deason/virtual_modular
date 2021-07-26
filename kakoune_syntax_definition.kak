hook global BufCreate .*[.]synth  %{
  set-option buffer filetype synth
}

provide-module synthsyntax %&
    add-highlighter shared/synthsyntax regions

    # A region from a `#` to the end of the line is a comment.
    add-highlighter shared/synthsyntax/ region '#' '\n' fill comment

    # Everything outside a region is a group of highlighters.
    add-highlighter shared/synthsyntax/other default-region group

    # Spaces are errors, eventually I'll fix that.
    add-highlighter shared/synthsyntax/other/ \
        regex [\ ] 0:Error


    # Highlighting for numbers.
    add-highlighter shared/synthsyntax/other/ \
        regex \b((\+|-)?[0-9]+(\.[0-9]+)?) 1:value

    # Highlighting for node definitions.
    add-highlighter shared/synthsyntax/other/ \
        regex \b([^=\n]+)= 1:variable

    # Highlighting for node constructors
    add-highlighter shared/synthsyntax/other/ \
        regex \b[A-Z][^\(\[=,]+ 0:function

    # Highlighting for patches.
    add-highlighter shared/synthsyntax/other/ \
        regex ^(\()[^,]+,[^,]+,[^\n]+(\))$ 1:meta 2:meta
&

# When a window's `filetype` option is set to this filetype...
hook global WinSetOption filetype=synth %{
    # Ensure our module is loaded, so our highlighters are available
    require-module synthsyntax

    # Link our higlighters from the shared namespace
    # into the window scope.
    add-highlighter window/synthsyntax ref synthsyntax

    # Add a hook that will unlink our highlighters
    # if the `filetype` option changes again.
    hook -once -always window WinSetOption filetype=.* %{
        remove-highlighter window/synthsyntax
    }
}

# Lastly, when a buffer is created for a new or existing file,
# and the filename ends with `.example`...
hook global BufCreate .+\.example %{
    # ...we recognise that as our filetype,
    # so set the `filetype` option!
    set-option buffer filetype example
}
