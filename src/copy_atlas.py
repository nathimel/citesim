import os
import sys

from sciterra import Atlas
from sciterra.vectorization import vectorizers

from analysis.plot import atlas_to_measurements
from analysis.plot import search_converged_ids

# We don't use the hydra.compose api, since we can't use sweeps with that anyways. Instead, we literally build a giant dataframe of all outputs in multirun.

atlas_dirs = {
    "Physics": "outputs/librarian=S2/vectorizer=SciBERT/center=hafenLowredshiftLymanLimit2017",
    "Linguistics": "outputs/librarian=S2/vectorizer=SciBERT/center=Imeletal2022",
    "Medicine": "outputs/librarian=S2/vectorizer=SciBERT/center=Torres2013",
    "Education": "outputs/librarian=S2/vectorizer=SciBERT/center=Ololube2012",
    "Philosophy": "outputs/librarian=S2/vectorizer=SciBERT/center=Bacon2019",
    "Economics": "outputs/librarian=S2/vectorizer=SciBERT/center=West2003",
    "Materials Science": "outputs/librarian=S2/vectorizer=SciBERT/center=Miele2022",
    "Geology": "outputs/librarian=S2/vectorizer=SciBERT/center=ForeroOrtega2021",
    "Mathematics": "outputs/librarian=S2/vectorizer=SciBERT/center=Andre2018",
}

from sciterra import Cartographer
from sciterra.vectorization.vectorizer import Vectorizer


def copy_atlas(atl: Atlas, vectorizer: Vectorizer, **kwargs) -> Atlas:
    """Create a new atlas containing the publications of an old one, and having a projection with a different vectorizer."""

    atl_new = Atlas(
        publications=list(atl.publications.values()),
    )
    crt = Cartographer(vectorizer=vectorizer)
    atl_new = crt.project(atl_new, **kwargs)
    return atl_new


def main():
    new_vectorizer = "Word2Vec"

    # Load the source atlas for the field
    field = "Mathematics"
    atl = Atlas.load(atlas_dirs[field])
    print(len(atl))

    save_dir = atlas_dirs[field].replace(
        "vectorizer=SciBERT", f"vectorizer={new_vectorizer}"
    )

    print(f"Copying atlas from {atlas_dirs[field]}")
    print(f"Projecting with {new_vectorizer}")

    # Create corpus for Word2Vec
    corpus_path = f"{save_dir}/corpus.txt"
    model_path = f"{save_dir}/w2v.model"
    overwrite = False  # make this an arg

    if not os.path.exists(corpus_path) or overwrite:
        abstracts = "\n".join([pub.abstract for pub in atl.publications.values()])
        with open(corpus_path, "w") as f:
            f.writelines(abstracts)

    # Initialize the vectorizer

    vectorizer = vectorizers[new_vectorizer](
        corpus_path=corpus_path,
        model_path=model_path,
    )

    atl_new = copy_atlas(atl, vectorizer, batch_size=32)

    atl_new.center = atl.center
    atl_new.history = atl.history
    atl_new.save(save_dir)

    # This step is necessary for word2vec which has failed embeddings

    # Take converged pubs that are in atlas
    converged_pub_ids = search_converged_ids(
        atl,
        num_pubs_added=1000,
    )
    converged_pub_ids = [id for id in converged_pub_ids if id in atl_new.ids]

    # measure with diff vectorizer
    df = atlas_to_measurements(
        atl_new,
        vectorizer=vectorizer,
        converged_pub_ids=converged_pub_ids,
        fields_of_study=[field],
    )

    # Save
    save_fn = "all_data.csv"
    save_fn = os.path.join(save_dir, save_fn)

    df.to_csv(save_fn, index=False)


if __name__ == "__main__":
    main()
