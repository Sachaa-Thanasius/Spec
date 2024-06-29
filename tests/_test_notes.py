import spec


class Inner(spec.Model):
    c: int


class Middle(spec.Model):
    b: Inner


class Outer(spec.Model):
    a: Middle


valid_sample = {"a": {"b": {"c": 1}}}
invalid_sample = {"a": {"b": {"c": "1"}}}

Outer(valid_sample)
Outer(invalid_sample)
