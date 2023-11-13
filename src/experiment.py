"""A simple class to contain initialized objects for a computational experiment measuring citation-similarity relationships."""

import os
import hydra
from omegaconf import DictConfig

from sciterra.mapping.cartography import Cartographer, iterate_expand
from sciterra.librarians import librarians
from sciterra.vectorization import vectorizers
from sciterra.mapping.atlas import Atlas

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


        ######################################################################
        # Initialize cartography tools
        ######################################################################

        # Get librarian
        librarian = librarians[config.experiment.librarian.name]

        # Get vectorizer
        vectorizer = vectorizers[config.experiment.vectorizer.name]
        # Get vectorizer kwargs if they are not null in config
        v_kwargs = {k:v for k,v in config.experiment.vectorizer.kwargs.items() if v is not None}

        self.cartographer = Cartographer(
            librarian=librarian(),
            vectorizer=vectorizer(
                **v_kwargs,
            ),
        )

        ######################################################################
        # Initialize/Load Atlas
        ######################################################################
        get_fullpath = lambda relpath: os.path.join(hydra.utils.get_original_cwd(), relpath)

        # Load
        self.atlas_dir = os.getcwd() # leaf dir
        atl = Atlas.load(self.atlas_dir)
        if len(atl):
            print(
                f"Loaded atlas has {len(atl)} publications and {len(atl.projection) if atl.projection is not None else 'None'} embeddings.\n"
            )
        else:
            print(f"Initializing atlas.")

            # Get the bibtex file containing the seed publication
            bibtex_fp = get_fullpath(config.filepaths.atlas_center_bibtex)

            # Get center from file
            atl_center = self.cartographer.bibtex_to_atlas(bibtex_fp)

            num_entries = len(atl_center.publications.values())
            if num_entries > 1:
                raise Exception(f"To build out a centered atlas, the center is specified by loading a bibtex file with a single entry. Found {num_entries} entries in {bibtex_fp}")

            atl = atl_center

        self.atlas = atl
        self.atlas.save(atlas_dirpath=self.atlas_dir)


    def expand_atlas(self) -> None:
        """Start or continue the expansion of an Atlas defined by the Experiment.
        """
        crt = self.config.experiment.cartography

        iterate_expand(
            atl=self.atlas,
            crt=self.cartographer,
            atlas_dir=self.atlas_dir,
            target_size=crt.target_size,
            max_failed_expansions=crt.max_failed_expansions,
            center=self.atlas.center,
            n_pubs_max=crt.n_pubs_max,
            call_size=self.config.experiment.librarian.call_size,
            record_pubs_per_update=True,
    )    
