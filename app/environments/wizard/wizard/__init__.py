from gym.envs.registration import register

register(
    id='Wizard-v0',
    entry_point='wizard.envs:WizardEnv',
)

