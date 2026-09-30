import io
import re

P = r"C:\Users\mitansh\Desktop\KAGGLE HACKATHONS\Predicting-Soil-Grain-Size-Distributions-from-Images\Model 7\Approach A\check_spatial_features.py"
src = io.open(P, encoding="utf-8", newline="").read()
lines = src.split("\n")

# (line_no_1based, [replacement lines])
repl = {
    783: ['        check("hand-built lag %d: one direction at %d pairs, the other above the floor"'
          ' % (lag, n_common),',
          '              counts[0] == n_common and counts[1] >= sf.MIN_PAIRS_PER_LAG, str(counts))'],
}
print("target line 783:", repr(lines[782]))
for i in (829, 847, 898, 912, 1029, 1071, 1152, 1190, 1197, 1312, 1521, 1529,
          1573, 1574, 1575, 1652, 1653, 1663, 1717, 1732):
    print(i + 1, repr(lines[i]))
