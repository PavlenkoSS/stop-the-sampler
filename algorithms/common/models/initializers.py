from flax import linen as nn

HIDDEN_INITS = ("lecun_normal", "orthogonal")


def hidden_kernel_init(name: str):
    """The law the hidden ``Dense`` kernels of a network are drawn from.

    ``lecun_normal`` is Flax's default, and what every run before this option
    was drawn from.  ``orthogonal`` draws each kernel (semi-)orthogonal at gain 1,
    so every singular value is 1 and AdamO's ``R_iso`` is zero at init: the
    network starts isometric and the regularizer keeps it there, instead of
    spending the first steps pulling it towards isometry.

    The output layers keep their own near-zero init under either law, so the
    sampler still starts from the reference process.
    """
    match name:
        case "lecun_normal":
            return nn.initializers.lecun_normal()
        case "orthogonal":
            return nn.initializers.orthogonal()
        case _:
            raise ValueError(f"hidden_init must be one of {HIDDEN_INITS}, got {name!r}")
