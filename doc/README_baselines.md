# Baseline models

## DCASE 24

The following scripts use the configuration in file:

```
config/baseline/baseline_dcase24.config
```

Run the following command to create the conda environment:

```
bash src/baseline/create_env.sh
```

Run the following command to install DCASE baseline model:

```
bash src/baseline/install_baseline.sh
```

Run the following command to prepare DCASE model and data:

```
bash src/baseline/prepare_baseline.sh
```

Run the following command to test DCASE inference:

```
bash src/baseline/test_inference.sh
```

## CLAP

Regarding the google cloud instance, some relevant configuration used:
- Image: debian-11-bullseye-v20250114 (IMPORTANT for the Java Version requirements of the metrics)
- HW: CPU, no GPU

The following scripts use the configuration in file:

```
config/clap/baseline_clap.config
```

Run the following command to create the conda environment:

```
bash src/clap/create_env.sh
```

Run the following command to install CLAP:

```
bash src/clap/install_clap.sh
```

Run the following command to test CLAP inference (each script call provides a different output for the same input file):

```
bash src/clap/test_inference.sh
```