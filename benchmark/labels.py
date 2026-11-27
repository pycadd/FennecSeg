"""Structure groups of the released FennecSeg models (TotalSegmentator dataset mask names)."""
ABDOMEN14 = ["spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas",
             "adrenal_gland_right", "adrenal_gland_left", "esophagus", "duodenum", "aorta",
             "inferior_vena_cava", "portal_vein_and_splenic_vein"]
SPINE26 = ([f"vertebrae_C{i}" for i in range(1, 8)] + [f"vertebrae_T{i}" for i in range(1, 13)]
           + [f"vertebrae_L{i}" for i in range(1, 7)] + ["sacrum"])
THORAX7 = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right",
           "lung_lower_lobe_right", "trachea", "heart"]
TASKS = {"abdomen14": {"labels": ABDOMEN14}, "spine26": {"labels": SPINE26}, "thorax7": {"labels": THORAX7}}
