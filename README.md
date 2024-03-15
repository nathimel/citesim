# citesim

## Replicating experiments

TODO.

The relevant scripts to run are

`python src/build_atlas.py`

`python src/analyze.py`


For now, to replicate results, or continue expanding/analyzing, select the folder of results you want and look at the .hydra/overrides.yaml file. That will give you a list of commands (overrides) that need to be passed when calling the relevant script. For example `outputs/librarian=S2/vectorizer=SciBERT/center=hafenLowredshiftLymanLimit2017/.hydra/overrides.yaml` has no extra commands, because it is the default setting. Meanwhile, `multirun/librarian=S2/vectorizer=SciBERT/center=Ololube2012/.hydra/overrides.yaml` specifies a list of commands that should follow calling the script. That overrides.yaml file corresponds to the following command:

`python src/build_atlas.py -m experiment.vectorizer.name=GPT2 experiment.atlas.center=Ololube2012 experiment.cartography.required_pub_conditions.fields_of_study="[Education]" experiment.cartography.target_size=30000 experiment.cartography.n_pubs_max=500`

Notice the quotes around the list containing Medicine in one of the overrides.
