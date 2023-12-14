<!-- ABOUT THE PROJECT -->
## SAVAE

Survival Analysis model based on Variational Autoencoders. 

This repository provides:
* Necessary scripts to train SAVAE and the state-of-the-art models (CoxPH, DeepSurv and Deephit) using [Pycox](https://github.com/havakv/pycox) package.
* Pre-processed and ready-to-use datasets included.
* Validation metrics (C-index and IBS) adapted from PyCox.
* A script to generate result tables as presented in the paper.
* Pre-trained models to save you training time.

For more details, see full paper [here]().


<!-- GETTING STARTED -->
## Getting Started
Follow these simple steps to make this project work on your local machine.

### Prerequisites
You should have the following installed on your machine:

* Ubuntu
* Python 3.8.0
* Packages in requirements.txt
  ```sh
  pip install -r requirements.txt
  ```

### Installation

Download the repo manually (as a .zip file) or clone it using Git.
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```

**IMPORTANT:** If you want to change the root folder name (named savae by default), you should also change the project_name variable in _utils.py_.

Download SAVAE and SOTA models weights and dictionaries with results from [here]() and place them in /savae/survival_analysis/.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

You can specify different configurations or training parameters in utils.py for both SAVAE and State-Of-The-Art (SOTA) models. 

To preprocess data, run the following command:
   ```sh
   python survival_analysis/preprocess_data.py
   ```

To train/test SAVAE and show results, run the following command:
   ```sh
   python survival_analysis/main_savae.py
   ```
To train/test SOTA models and show results, run the following command:
   ```sh
   python survival_analysis/main_sota.py
   ```

If you want to display comparison results between SAVAE and SOTA models, run the following command:
   ```sh
   python survival_analysis/results_display.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



[//]: # (<!-- LICENSE -->)

[//]: # (## License)

[//]: # ()
[//]: # (Distributed under the XXX License. See `LICENSE.txt` for more information.)

[//]: # ()
[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)



<!-- CONTACT -->
## Contact

Patricia A. Apellaniz - patricia.alonsod@upm.es

<p align="right">(<a href="#readme-top">back to top</a>)</p>


[//]: # (<!-- ACKNOWLEDGMENTS -->)

[//]: # (## Acknowledgments)

[//]: # ()
[//]: # (* []&#40;&#41;)

[//]: # (* []&#40;&#41;)

[//]: # (* []&#40;&#41;)

[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)

