from gym.envs.registration import register
from .KSS.Env import KSSEnv
from .KSS.EnvMulti import KSSEnvMulti
from .KES.Env import KESEnv
from .KES.EnvCo import KESEnvCo

register(
    id='KSS-v0',
    entry_point='Scripts.Envs.KSS.Env:KSSEnv',
)
register(
    id='KSS-v2',
    entry_point='Scripts.Envs.KSS.EnvMulti:KSSEnvMulti',
)
