from columnmap.core.settings import env_overrides, merge, normalize, validate


# the same checks for a dict already in memory (tests only); env overrides apply on top
def parse(data: dict, env=None) -> dict:
    env = {} if env is None else env
    return validate(merge(normalize(data), env_overrides(env)))
