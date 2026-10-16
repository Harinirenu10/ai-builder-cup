# SMPL Model Files

The SMPL body model is license-gated by the Max Planck Institute for Intelligent Systems.
Due to licensing terms, SMPL model files cannot be redistributed in public source repositories.

## Required Setup:
1. Register and agree to the license terms at: https://smpl.is.tue.mpg.de
2. Download the SMPL model package (`SMPL_python_v.1.1.0.zip` or neutral/female/male pkl models).
3. Extract and place the model files in this directory:
   - `services/geometry/models/smpl/basicModel_neutral_lbs_10_207_0_v1.1.0.pkl` (or `SMPL_NEUTRAL.pkl`)
   - `services/geometry/models/smpl/basicModel_f_lbs_10_207_0_v1.1.0.pkl` (or `SMPL_FEMALE.pkl`)
   - `services/geometry/models/smpl/basicModel_m_lbs_10_207_0_v1.1.0.pkl` (or `SMPL_MALE.pkl`)

Note: In tests and local dry-runs, a mock/synthetic mesh generator is used when license-gated files are absent.
