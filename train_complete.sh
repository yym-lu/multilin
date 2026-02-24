

# 包含 config1 到 config7 的所有模型
# 每个模型使用 5 个随机种子进行训练，计算平均值，并生成图表

# 定义所有模型配置
# 格式: "config_name model_name"
MODEL_CONFIGS=(
    "config1 colda_8k"
    "config2 colda_4k"
    "config3 colda_20k"
    "config4 olid_13k"
    "config5 colda_4k_olid_13k"
    "config6 colda_20k_olid_13k"
    "config7 colda_8k_olid_13k"
)

# 定义使用的种子
SEEDS=(42 43 44)

# 基础路径
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${BASE_DIR}/models"

# 创建模型保存目录
mkdir -p "${MODELS_DIR}"

# 遍历所有模型配置
for model in "${MODEL_CONFIGS[@]}"; do
    read -r config model_name <<< "$model"
    
    echo -e "\n========================================"
    echo "正在处理模型: ${model_name} (配置: ${config})"
    echo "========================================"
    
    MODEL_SAVE_PATH="${MODELS_DIR}/${model_name}"
    mkdir -p "${MODEL_SAVE_PATH}"
    
    # 1. 训练每个种子
    for SEED in "${SEEDS[@]}"; do
        SEED_DIR="${MODEL_SAVE_PATH}/seed_${SEED}"
        
        # 检查是否已经训练过（检查 best_model 是否存在）
        if [ -d "${SEED_DIR}/best_model" ]; then
            echo "种子 ${SEED} 已存在，跳过训练..."
            continue
        fi
        
        echo "开始训练 seed=${SEED}..."
        
        python "${BASE_DIR}/train.py" \
            --config "${config}" \
            --save_path "${SEED_DIR}" \
            --seed "${SEED}"
            
        if [ $? -ne 0 ]; then
            echo "错误: 训练失败 (model=${model_name}, seed=${SEED})"
            exit 1
        fi
    done
    
    # 2. 计算平均结果
    echo "计算平均结果..."
    python "${BASE_DIR}/calculate_average_results.py" "${MODEL_SAVE_PATH}"
    
    # 3. 生成训练曲线
    echo "生成训练曲线..."
    python "${BASE_DIR}/plot_metrics.py" --model_names "${model_name}" --models_dir "${MODELS_DIR}"
    
    echo "模型 ${model_name} 处理完成！"
done

echo -e "\n========================================"
echo "所有模型训练完成！开始进行测试和绘图..."
echo "========================================"

# 4. 测试模型并更新可视化图表
echo "测试所有模型..."
python "${BASE_DIR}/test_models.py"

echo "更新可视化图表..."
python "${BASE_DIR}/plot_gain_cost.py"

echo "所有任务全部完成！"
