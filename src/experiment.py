"""A simple class to contain initialized objects for a computational experiment measuring citation-similarity relationships."""

import os
import hydra
from omegaconf import DictConfig

from sciterra import Atlas
from sciterra.vectorization.vectorizer import Vectorizer
from sciterra.mapping.tracing import AtlasTracer
from sciterra.mapping.cartography import pub_has_attributes, pub_has_fields_of_study

class Experiment:
    def __init__(
        self,
        config: DictConfig,
        ) -> None:
        """Construct an experiment object, which can contain all the necessary data for running the computational experiment measuring citation-similarity relationships.

        Args:
            config: a Hydra config

        """
        self.config = config

        get_fullpath = lambda relpath: os.path.join(hydra.utils.get_original_cwd(), relpath)

        # Load
        self.atlas_dir = os.getcwd()
        self.bibtex_fp = get_fullpath(config.filepaths.atlas_center_bibtex)

        self.tracer = AtlasTracer(
            atlas_dir = self.atlas_dir,
            atlas_center_bibtex = self.bibtex_fp,

            librarian_name = config.experiment.librarian.name,
            vectorizer_name = config.experiment.vectorizer.name,

            librarian_kwargs = {k:v for k,v in config.experiment.librarian.kwargs.items() if v is not None},
            vectorizer_kwargs = {k:v for k,v in config.experiment.vectorizer.kwargs.items() if v is not None},
        )

    def expand_atlas(self) -> None:
        """Start or continue the expansion of an Atlas defined by the Experiment.
        """
        crt = self.config.experiment.cartography

        # Get Cartographer.project kwargs
        rpcs = crt.required_pub_conditions
        require_func = lambda pub: (
            pub_has_attributes(pub, rpcs.attributes) 
            and pub_has_fields_of_study(pub, rpcs.fields_of_study)
        )

        call_size = crt.call_size if hasattr(crt, "call_size") else None

        self.tracer.expand_atlas(
            target_size = crt.target_size,
            max_failed_expansions = crt.max_failed_expansions,
            n_pubs_max = crt.n_pubs_max,
            call_size = call_size,
            record_pubs_per_update=True,
            require_func=require_func,
            batch_size=crt.batch_size,
        )

    @property
    def atlas(self) -> Atlas:
        return self.tracer.atlas
    
    @property
    def vectorizer(self) -> Vectorizer:
        return self.tracer.cartographer.vectorizer