from gym.envs.registration import register

register(
    id='BasicTrickGame-v0',
    entry_point='basictrickgame.envs:BasicTrickGameEnv',
)