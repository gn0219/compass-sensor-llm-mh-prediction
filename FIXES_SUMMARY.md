# Fixes Summary

## Issues Addressed

### 1. Sensor Features Missing for Some Users (Issue #1)
**Problem**: Some users had no sensor features data, causing errors during prompt generation.

**Solution**: Modified `sample_multiinstitution_testset()` in `src/sensor_transformation.py` to:
- Pre-filter users by checking if they have sensor data before their last N EMA samples
- Only select users with sufficient sensor data history
- Lines: 1194-1218

### 2. Removed Unnecessary Config Parameters from Evaluation Header (Issue #2)
**Problem**: Evaluation configuration displayed adaptive/immediate window settings that weren't relevant for GLOBEM dataset.

**Solution**: Modified `run_evaluation.py` to:
- Simplified evaluation header to show only `Format` and `Test Filter` settings
- Removed window/mode/adaptive/immediate parameters from display
- Lines: 110-112

### 3. Fixed FCTCI Feature Names and Format (Issue #3)
**Problem**: FCTCI format wasn't matching the paper's markdown table format with correct feature names.

**Solution**: Modified `features_to_text_fctci()` in `src/sensor_transformation.py` to:
- Added proper feature name mapping (e.g., 'total_distance_traveled(meters)', 'time_at_home(minutes)')
- Fixed markdown table format (no separator row, proper pipe delimiters)
- Lines: 494-576

### 4. Fixed Health-LLM Format to Match Paper (Issue #4)
**Problem**: Health-LLM format wasn't matching the exact wording from the paper.

**Solution**: Modified `features_to_text_healthllm()` in `src/sensor_transformation.py` to:
- Changed to directly read 14dhist features (already aggregated statistics)
- Match exact paper wording: "The recent 14-days sensor readings show: [Steps] is {value}. [Sleep] efficiency, duration the user stayed in bed after waking up, duration the user spent to fall asleep, duration the user stayed awake but still in bed, duration the user spent to fall asleep are {vals} mins in average"
- Added steps feature to health-llm config
- Lines: 579-675

### 5. Fixed Unicode/Emoji Encoding Issues
**Problem**: Windows console (cp949) couldn't encode emoji characters, causing crashes.

**Solution**: Replaced all emojis in `src/evaluation_runner.py` with ASCII brackets:
- ✅ → [OK]
- ❌ → [ERROR]
- ⚠️ → [WARNING]
- 🔄 → [RESUME]
- 📊 → [Data]
- 🎨 → [Prompt]
- 📚 → [ICL]
- 🤖 → [Prompt]
- 💾 → [Checkpoint]

### 6. Fixed ICL Example Selection for Multi-Institution Testset (Issue #6)
**Problem**: Personalized and hybrid ICL modes couldn't find examples because only testset samples were included in lab_df.

**Solution**: Modified `sample_multiinstitution_testset()` to:
- Include FULL label dataframe (all samples from all institutions)
- Mark testset samples with `is_testset` column
- evaluation_runner filters to testset for prediction, but uses full data (excluding testset) for ICL selection
- Lines: 1212, 1227-1229, 1252-1257

**Files Modified**:
- `src/sensor_transformation.py`: Lines 1208-1216, 1223-1229
- `src/evaluation_runner.py`: Lines 11 (added pandas import), 482-494 (filter testset), 589 (use filtered lab_df for ICL)
- `config/globem_use_cols.json`: Line 104 (added steps feature to health-llm)

## Testing
Run with: `python run_evaluation.py --mode batch --save-prompts-only --seed 42 --n_shot 4 --source hybrid --selection random`

Expected:
- 414 samples from multi-institution testset (65+28+45 users * 3 samples each)
- No unicode errors
- ICL examples successfully selected for personalized/hybrid modes
- Proper formatting for fctci and health-llm modes


