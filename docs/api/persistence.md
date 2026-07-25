# Persistence

Three free functions and nothing else (ADR-0072): the library hands back bytes
and takes bytes; where those bytes live is entirely the user's.

```{eval-rst}
.. autofunction:: dataforge_ml._serialization.serialize
```

```{eval-rst}
.. autofunction:: dataforge_ml._serialization.deserialize
```

```{eval-rst}
.. autofunction:: dataforge_ml._serialization.inspect
```

```{eval-rst}
.. autoclass:: dataforge_ml._serialization.IncompatibleArtifactError
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: dataforge_ml._serialization.ArtifactPythonVersionWarning
   :members:
   :undoc-members:
   :show-inheritance:
```
