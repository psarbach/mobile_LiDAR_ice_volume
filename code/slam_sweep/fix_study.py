#cat > fix_study.py << 'EOF'
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

storage = "sqlite:////home/psarbach/glim_sweep/runs/glim_sweep.db"
old = optuna.load_study(study_name="glim_sweep", storage=storage)

new = optuna.create_study(
    study_name="glim_sweep_v2",
    storage=storage,
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)

good = [t for t in old.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value < 99.0]
for t in good:
    new.add_trial(t)
print(f"Copied {len(good)} good trials into glim_sweep_v2")