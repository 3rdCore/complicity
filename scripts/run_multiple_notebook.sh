#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ==========================================
# 1. Define Which Configurations to Run
# ==========================================
# List the names of the configurations you want to execute here.
# e.g., "config1 config2"
CONFIGS_TO_RUN="config1 config2 config3"

# ==========================================
# 2. Shared/Default Constants
# ==========================================
# Parameters common to ALL configurations can be defined here.
SCRIPTS="main"
LOAD_PRETRAINEDS="False"
NETWORK_NAMES="MLP"
OPTIMIZER_NAMES="adamw"
WEIGHT_DECAYS="0.0001"
DEBUG_MODES="False"
N_OUTPUTS_LIST="256"
SEEDS="5"
ATTR_PROBS="0.5"
CMNIST_DIGITS_PER_CLASSES="5"  # default; can be overridden per config

# Initialize timestamp once so all results go into the same main folder
result_timestamp=$(date +%Y%m%d-%H%M%S)
if [ -n "$1" ]; then
  result_timestamp="${result_timestamp}_${1}"
fi
echo "Results will be saved under results/${result_timestamp}/"

# ==========================================
# 3. Main Configuration Loop
# ==========================================

for config_name in $CONFIGS_TO_RUN; do
  echo "----------------------------------------------------------------"
  echo "Loading Configuration: ${config_name}"
  echo "----------------------------------------------------------------"

  # Reset all config-specific variables to prevent leakage between configs
  EXPERIMENT_SETTINGS=""
  SPUR_PROBS_SETTING1="" FLIP_PROBS_SETTING1="" ENV_NOISINESSES_SETTING1="" UNINFORMATIVE_MAJORITYS=""
  SPUR_PROBS_SETTING2="" FLIP_PROBS_SETTING2="" ENV_NOISINESSES_SETTING2="" WATERMARK_BANK_SIZES=""
  _CONFIG_DIGITS_PER_CLASS=""  # per-config override for CMNIST_DIGITS_PER_CLASSES

  # Define specific parameters based on the configuration name
  case $config_name in

    "config1")
      # --- CONFIG 1 START ---
      EXPERIMENT_SETTINGS="2"

      # Setting 2 params
      SPUR_PROBS_SETTING2="0.5"
      FLIP_PROBS_SETTING2="$(seq 0.0 0.02 0.3)" #sweeping over FLIP_PROBS
      ENV_NOISINESSES_SETTING2="0.0"
      WATERMARK_BANK_SIZES="200"
      # _CONFIG_DIGITS_PER_CLASS="5"  # uncomment to override per config
      # --- CONFIG 1 END ---
      ;;
    "config2")
      # --- CONFIG 2 START ---
      EXPERIMENT_SETTINGS="1"

      # Setting 1 params
      SPUR_PROBS_SETTING1="$(seq 0 0.02 0.3)" #sweeping over SPUR_PROBS
      FLIP_PROBS_SETTING1="0.0"
      UNINFORMATIVE_MAJORITYS="False"
      ENV_NOISINESSES_SETTING1="0.0"
      ;;
    "config3")
      # --- CONFIG 3 START ---
      EXPERIMENT_SETTINGS="2"

      # Setting 2 params
      SPUR_PROBS_SETTING2="0.5"
      FLIP_PROBS_SETTING2="0.15"
      ENV_NOISINESSES_SETTING2="0.0"
      WATERMARK_BANK_SIZES=$(seq 0 0.25 2.5 | awk '{printf "%.0f\n", 10^$1}' | sort -nu) 
      # _CONFIG_DIGITS_PER_CLASS="5"  # uncomment to override per config
      ;;
    *)
      echo "Error: Configuration '$config_name' not found."
      exit 1
      ;;
  esac

  # Apply per-config digits_per_class override (or fall back to shared default)
  if [ -n "$_CONFIG_DIGITS_PER_CLASS" ]; then
    CMNIST_DIGITS_PER_CLASSES="$_CONFIG_DIGITS_PER_CLASS"
  fi

  # ==========================================
  # 4. Job Submission Loops
  # ==========================================
  # (Loops iterate over the parameters currently loaded above)

  for script in $SCRIPTS; do
    for load_pretrained in $LOAD_PRETRAINEDS; do
      for network_name in $NETWORK_NAMES; do
        for optimizer_name in $OPTIMIZER_NAMES; do
        for weight_decay in $WEIGHT_DECAYS; do
          for debug_mode in $DEBUG_MODES; do
            for n_outputs in $N_OUTPUTS_LIST; do
              for experiment_setting in $EXPERIMENT_SETTINGS; do
                for attr_prob in $ATTR_PROBS; do
                  for digits_per_class in $CMNIST_DIGITS_PER_CLASSES; do
                    for seed in $SEEDS; do

                      # Select setting-specific parameters
                      if [ "$experiment_setting" = "1" ]; then
                        spur_probs="$SPUR_PROBS_SETTING1"
                        flip_probs="$FLIP_PROBS_SETTING1"
                        env_noisinesses="$ENV_NOISINESSES_SETTING1"
                        setting_specific_list="$UNINFORMATIVE_MAJORITYS"
                      else
                        spur_probs="$SPUR_PROBS_SETTING2"
                        flip_probs="$FLIP_PROBS_SETTING2"
                        env_noisinesses="$ENV_NOISINESSES_SETTING2"
                        setting_specific_list="$WATERMARK_BANK_SIZES"
                      fi

                      for env_noisiness in $env_noisinesses; do
                        for spur_prob in $spur_probs; do
                          for flip_prob in $flip_probs; do
                            for setting_param in $setting_specific_list; do

                              # Parse setting-specific parameter
                              if [ "$experiment_setting" = "1" ]; then
                                uninformative_majority="$setting_param"
                                watermark_bank_size=""
                              else
                                uninformative_majority=""
                                watermark_bank_size="$setting_param"
                              fi

                              # Construct experiment name
                              # Added config_name to exp_name so results don't overwrite if params overlap
                              exp_name="${config_name}_setting${experiment_setting}_${network_name}_spur${spur_prob}_flip${flip_prob}_env_noise${env_noisiness}"
                              
                              if [ $(echo "$N_OUTPUTS_LIST" | wc -w) -gt 1 ]; then exp_name="${exp_name}_n${n_outputs}"; fi
                              if [ $(echo "$WEIGHT_DECAYS" | wc -w) -gt 1 ]; then exp_name="${exp_name}_wd${weight_decay}"; fi
                              if [ $(echo "$OPTIMIZER_NAMES" | wc -w) -gt 1 ]; then exp_name="${exp_name}_opt${optimizer_name}"; fi
                              if [ $(echo "$ATTR_PROBS" | wc -w) -gt 1 ]; then exp_name="${exp_name}_attr${attr_prob}"; fi
                              if [ $(echo "$CMNIST_DIGITS_PER_CLASSES" | wc -w) -gt 1 ]; then exp_name="${exp_name}_digits${digits_per_class}"; fi
                              
                              # Setting-specific naming
                              if [ "$experiment_setting" = "1" ]; then
                                exp_name="${exp_name}_uninf${uninformative_majority}"
                              else
                                exp_name="${exp_name}_bank${watermark_bank_size}"
                              fi
                              if [ $(echo "$LOAD_PRETRAINEDS" | wc -w) -gt 1 ]; then exp_name="${exp_name}_pre${load_pretrained}"; fi
                              if [ $(echo "$DEBUG_MODES" | wc -w) -gt 1 ]; then exp_name="${exp_name}_debug${debug_mode}"; fi

                              folder="${result_timestamp}/${exp_name}/${seed}"
                              
                              # Create sbatch script
                              job_name="${exp_name}_${seed}"
                              job_script="run_${script}_job_${job_name}.sh"

                              cat <<EOF > $job_script
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${PROJECT_ROOT}/results/${folder}/${script}-%j.out
#SBATCH --error=${PROJECT_ROOT}/results/${folder}/${script}-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=long

module load cuda/11.8
cd ${PROJECT_ROOT}
source .venv/bin/activate
KERNEL_NAME="invariant-bench"
export EXPERIMENT_SETTING=$experiment_setting
export ATTR_PROB=$attr_prob
export SPUR_PROB=$spur_prob
export FLIP_PROB=$flip_prob
export ENV_NOISINESS=$env_noisiness
export CMNIST_DIGITS_PER_CLASS=$digits_per_class
export UNINFORMATIVE_MAJORITY=$uninformative_majority
export WATERMARK_BANK_SIZE=$watermark_bank_size
export SEED=$seed
export RESULT_FOLDER=$folder
export LOAD_PRETRAINED=$load_pretrained
export NETWORK_NAME=$network_name
export OPTIMIZER_NAME=$optimizer_name
export WEIGHT_DECAY=$weight_decay
export DEBUG_MODE=$debug_mode
export N_OUTPUTS=$n_outputs

mkdir -p ${PROJECT_ROOT}/results/\${RESULT_FOLDER}
${PROJECT_ROOT}/.venv/bin/jupyter nbconvert --to notebook --execute ${script}.ipynb --ExecutePreprocessor.kernel_name=${KERNEL_NAME} --ExecutePreprocessor.timeout=0 --output-dir ${PROJECT_ROOT}/results/\${RESULT_FOLDER} --output executed_${script}_notebook.ipynb
EOF
                              mkdir -p ${PROJECT_ROOT}/results/${folder}
                              sbatch $job_script
                              rm -f $job_script
                            
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
        done
      done
    done
  done
done # End of Config Loop

echo "All notebook jobs submitted!"