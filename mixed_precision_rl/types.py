from jaxtyping import Array, Bool, Float, Int, PyTree

Action = Float[Array, "... action_dim"]
Observation = Float[Array, "... obs_dim"]
Reward = Float[Array, "... 1"]
Done = Bool[Array, "... 1"]
Value = Float[Array, "... 1"]
LogProb = Float[Array, "... 1"]

EpisodeStarted = Bool[Array, "... 1"]
EpisodeSteps = Int[Array, "... 1"]
EpisodeReturns = Float[Array, "... 1"]

EnvState = PyTree
EnvParams = PyTree

type LogDict = dict[str, float | Float[Array, ""]]
