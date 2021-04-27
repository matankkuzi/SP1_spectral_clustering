<div style="text-align: center;"><img src="https://mma.prnewswire.com/media/1167265/Tel_Aviv_University_Logo.jpg?p=facebook" width="50%" height="50%" alt="TAU"></div>

# TAU Software Project Course
## Final Project
### Submitted by
* **Matan Kuzi** &ensp;
* **Dana Petrank**&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; 

### Project Modules
#### K-means C-API
The following files are required for running the K-means++ algorithm (using Python's C-API)
* `kmeans.c`
* `matoperations.c` `matoperations.h`
* `kmeans_pp.py`
* `setup.py`


#### Spectral Clustering Algorithm
* `main.py` — project's entering point
* `spectral_funcs.py` — Spectral Clustering algorithm implementation
* `kmeans_pp.py` - K-means++ algorithm implementation
* `input_output_funcs.py` — user communication: parsing user arguments, genarating data and output result files
* `tasks.py` — invoke tasks


### How to run?
Use the following command:
```bash
python3.8.5 -m invoke run -k [k] -n [n] [--[no-]Random]
```

Output files:
* `data.txt` — randomly generated data;
* `clusters.txt` — computed clusters for both algorithms (assignments);
* `clusters.pdf` — graphical visualization comparing K-means and Spectral Clustering algorithms.



