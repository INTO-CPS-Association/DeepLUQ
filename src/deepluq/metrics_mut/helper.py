#!/usr/bin/env python
# -*- coding: utf-8 -*-


def print_metric(match_metrics, miss_metrics, ghost_metrics):
    # Header Format
    # 1. Print the Header Row
    print("-" * 110)
    print(
        f"{'Obj':<4} {'Lbl':<4} {'Rate':<5} | "
        f"{'VR_O':<7} {'IE_O':<7} {'MI_O':<7} {'Var_O':<7} {'PS_O':<7} | "
        f"{'VR_M':<7} {'IE_M':<7} {'MI_M':<7} {'Var_M':<7} {'PS_M':<7} | "
        f"{'IoU':<6}"
    )
    print("-" * 110)

    # 3. Loop through results and print values
    for m in match_metrics:
        print(
            f"{m['id']:<4} "
            f"{m['label']:<4} "
            f"{m['detect_rate (out of 10)']:<5.1f} | "

            # Original Metrics
            f"{m['vr_orig']:<7.4f} "
            f"{m['ie_orig']:<7.4f} "
            f"{m['mi_orig']:<7.4f} "
            f"{m['var_orig']:<7.4f} "
            f"{m['ps_orig']:<7.4f} | "

            # Mutant Metrics
            f"{m['vr_mut']:<7.4f} "
            f"{m['ie_mut']:<7.4f} "
            f"{m['mi_mut']:<7.4f} "
            f"{m['var_mut']:<7.4f} "
            f"{m['ps_mut']:<7.4f} | "

            # Raw Accuracy Stats
            f"{m['avg_iou']:<6.4f} "
        )

    print("-" * 110)

    # 1. Print Header
    print("-" * 72)
    print(f"{'Obj':<5} {'Lbl':<5} {'MissRt':<8} {'IE':<10} {'VR':<10} {'MI':<10} {'PS':<10} {'Var':<10}")
    print("-" * 72)

    # 2. Print Rows
    for m in miss_metrics:
        print(
            f"{m['id']:<5} "
            f"{m['label']:<5} "
            f"{m['miss_rate (out of 10)']:<8.1f} "
            f"{m['ie_miss']:<10.4f} "
            f"{m['vr_miss']:<10.4f} "
            f"{m['mi_miss']:<10.4f} "
            f"{m['ps_miss']:<10.4f} "
            f"{m['var_miss']:<10.4f}"
        )

    print("-" * 72)

    # 1. Print Header
    print("-" * 72)
    print(f"{'Obj':<5} {'Lbl':<5} {'GhostRt':<8} {'IE':<10} {'VR':<10} {'MI':<10} {'PS':<10} {'Var':<10}")
    print("-" * 72)

    # 2. Print Rows
    for m in ghost_metrics:
        print(
            f"{m['id']:<5} "
            f"{m['label']:<5} "
            f"{m['ghost_rate (out of 10)']:<8.1f} "
            f"{m['ie_ghost']:<10.4f} "
            f"{m['vr_ghost']:<10.4f} "
            f"{m['mi_ghost']:<10.4f} "
            f"{m['ps_ghost']:<10.4f} "
            f"{m['var_ghost']:<10.4f}"
        )

    print("-" * 72)


def init_metrics():
    return {'[img_level] iskill_miss_p_value': [],
            '[img_level] iskill_miss_cohens_h': [],
            '[img_level] iskill_miss_power': [],
            '[img_level] iskill_miss_count': [],

            '[img_level] iskill_ghost_p_value': [],
            '[img_level] iskill_ghost_cohens_h': [],
            '[img_level] iskill_ghost_power': [],
            '[img_level] iskill_ghost_count': [],

            '[img_level] iskill_miss_ghost_p_value': [],
            '[img_level] iskill_miss_ghost_cohens_h': [],
            '[img_level] iskill_miss_ghost_power': [],
            '[img_level] iskill_miss_ghost_count': [],

            '[obj_level] kill_rate_miss_mean': [],
            '[obj_level] kill_rate_miss_std': [],

            '[obj_level] kill_rate_ghost_mean': [],
            '[obj_level] kill_rate_ghost_std': [],

            '[obj_level] kill_rate_miss_ghost_mean': [],
            '[obj_level] kill_rate_miss_ghost_std': [],

            '[match] iou_mean': [],
            '[match] match_rate': [],
            '[match] vr_diff': [],
            '[match] ie_diff': [],
            '[match] mi_diff': [],
            '[match] var_diff': [],
            '[match] ps_diff': [],

            '[miss] miss_rate': [],
            '[miss] vr_diff': [],
            '[miss] ie_diff': [],
            '[miss] mi_diff': [],
            '[miss] var_diff': [],
            '[miss] ps_diff': [],

            '[ghost] ghost_rate': [],
            '[ghost] vr_diff': [],
            '[ghost] ie_diff': [],
            '[ghost] mi_diff': [],
            '[ghost] var_diff': [],
            '[ghost] ps_diff': []
            }


def update_metrics(metrics, iskill_miss, iskill_ghost, iskill_miss_ghost,
                   ms_obj_level, match_metrics, miss_metrics, ghost_metrics):
    # 1. Image Level Skill Checks
    metrics['[img_level] iskill_miss_p_value'].append(iskill_miss['p_value'])
    metrics['[img_level] iskill_miss_cohens_h'].append(iskill_miss['cohens_h'])
    metrics['[img_level] iskill_miss_power'].append(iskill_miss['power'])
    metrics['[img_level] iskill_miss_count'].append(iskill_miss['observed_success'])

    metrics['[img_level] iskill_ghost_p_value'].append(iskill_ghost['p_value'])
    metrics['[img_level] iskill_ghost_cohens_h'].append(iskill_ghost['cohens_h'])
    metrics['[img_level] iskill_ghost_power'].append(iskill_ghost['power'])
    metrics['[img_level] iskill_ghost_count'].append(iskill_ghost['observed_success'])

    metrics['[img_level] iskill_miss_ghost_p_value'].append(iskill_miss_ghost['p_value'])
    metrics['[img_level] iskill_miss_ghost_cohens_h'].append(iskill_miss_ghost['cohens_h'])
    metrics['[img_level] iskill_miss_ghost_power'].append(iskill_miss_ghost['power'])
    metrics['[img_level] iskill_miss_ghost_count'].append(iskill_miss_ghost['observed_success'])

    # 2. Object Level Kill Rates
    metrics['[obj_level] kill_rate_miss_mean'].append(ms_obj_level['[obj_level] kill_rate_miss_mean'])
    metrics['[obj_level] kill_rate_ghost_mean'].append(ms_obj_level['[obj_level] kill_rate_ghost_mean'])
    metrics['[obj_level] kill_rate_miss_ghost_mean'].append(ms_obj_level['[obj_level] kill_rate_miss_ghost_mean'])
    metrics['[obj_level] kill_rate_miss_std'].append(ms_obj_level['[obj_level] kill_rate_miss_std'])
    metrics['[obj_level] kill_rate_ghost_std'].append(ms_obj_level['[obj_level] kill_rate_ghost_std'])
    metrics['[obj_level] kill_rate_miss_ghost_std'].append(ms_obj_level['[obj_level] kill_rate_miss_ghost_std'])

    # 3. Match Metrics
    metrics['[match] iou_mean'].append(match_metrics['iou_ms'])
    metrics['[match] match_rate'].append(match_metrics['match_rate'])
    metrics['[match] vr_diff'].append(match_metrics['vr_ms'])
    metrics['[match] ie_diff'].append(match_metrics['ie_ms'])
    metrics['[match] mi_diff'].append(match_metrics['mi_ms'])
    metrics['[match] var_diff'].append(match_metrics['var_ms'])
    metrics['[match] ps_diff'].append(match_metrics['ps_ms'])

    # 4. Miss Metrics
    metrics['[miss] miss_rate'].append(miss_metrics['miss_rate'])
    metrics['[miss] vr_diff'].append(miss_metrics['vr_ms'])
    metrics['[miss] ie_diff'].append(miss_metrics['ie_ms'])
    metrics['[miss] mi_diff'].append(miss_metrics['mi_ms'])
    metrics['[miss] var_diff'].append(miss_metrics['var_ms'])
    metrics['[miss] ps_diff'].append(miss_metrics['ps_ms'])

    # 5. Ghost Metrics
    metrics['[ghost] ghost_rate'].append(ghost_metrics['ghost_rate'])
    metrics['[ghost] vr_diff'].append(ghost_metrics['vr_ms'])
    metrics['[ghost] ie_diff'].append(ghost_metrics['ie_ms'])
    metrics['[ghost] mi_diff'].append(ghost_metrics['mi_ms'])
    metrics['[ghost] var_diff'].append(ghost_metrics['var_ms'])
    metrics['[ghost] ps_diff'].append(ghost_metrics['ps_ms'])

    return metrics
