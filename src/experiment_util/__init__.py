import omegaconf

# fmt: off
omegaconf.OmegaConf.register_new_resolver("cond", lambda cond, val_true, val_false: val_true if cond else val_false)
omegaconf.OmegaConf.register_new_resolver("not", lambda bool1: not bool1)
omegaconf.OmegaConf.register_new_resolver("or", lambda bool1, bool2: bool1 or bool2)
omegaconf.OmegaConf.register_new_resolver("and", lambda bool1, bool2: bool1 and bool2)
omegaconf.OmegaConf.register_new_resolver("max", lambda a, b: max(a, b))
omegaconf.OmegaConf.register_new_resolver("min", lambda a, b: min(a, b))
omegaconf.OmegaConf.register_new_resolver("add", lambda a, b: a + b)
omegaconf.OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
omegaconf.OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
omegaconf.OmegaConf.register_new_resolver("div", lambda a, b: a / b)
omegaconf.OmegaConf.register_new_resolver("int_div", lambda a, b: a // b)
# fmt: on 