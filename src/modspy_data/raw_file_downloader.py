import requests
from urllib.parse import urlparse
import os
from tqdm import tqdm

# List of URLs
urls = [
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/alliance_gene_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/alliance_gene_to_phenotype_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/alliance_gene_to_phenotype_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/alliance_publication_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/alliance_publication_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/ctd_chemical_to_disease_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/ctd_chemical_to_disease_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/flybase_publication_to_gene_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/flybase_publication_to_gene_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/goa_go_annotation_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/goa_go_annotation_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/hgnc_gene_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/hgnc_gene_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/hpoa_disease_phenotype_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/hpoa_disease_phenotype_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/mgi_publication_to_gene_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/mgi_publication_to_gene_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/omim_gene_to_disease_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/omim_gene_to_disease_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/pombase_gene_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/pombase_gene_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/pombase_gene_to_phenotype_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/pombase_gene_to_phenotype_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/reactome_chemical_to_pathway_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/reactome_chemical_to_pathway_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/reactome_gene_to_pathway_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/reactome_gene_to_pathway_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/rgd_publication_to_gene_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/rgd_publication_to_gene_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/sgd_publication_to_gene_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/sgd_publication_to_gene_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/xenbase_gene_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/xenbase_gene_to_phenotype_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/xenbase_gene_to_phenotype_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/xenbase_publication_to_gene_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/xenbase_publication_to_gene_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/zfin_gene_to_phenotype_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/zfin_gene_to_phenotype_nodes.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/zfin_publication_to_gene_edges.tsv",
    "https://data.monarchinitiative.org/monarch-kg-dev/kgx/zfin_publication_to_gene_nodes.tsv"
]


def download_large_file(url):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raises an HTTPError for bad responses

            # Extracts the filename from the URL
            filename = os.path.basename(urlparse(url).path)
            
            # Total size in bytes.
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 100*8192  # 8KB
            num_bars = total_size // chunk_size

            # Write the file to the current directory in chunks
            with open(filename, 'wb') as f, tqdm(
                total=num_bars, unit='MB', unit_scale=True, desc=filename
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size): 
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(1)

            print(f"Downloaded {filename}")

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")

for url in urls:
    download_large_file(url)