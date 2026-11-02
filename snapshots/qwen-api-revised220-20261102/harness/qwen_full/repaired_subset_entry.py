"""Verified frozen bootstrap for the repaired native subset lifecycle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import repaired_subset_contract
import subset_entry
import perception_entry


def provider_configuration(path, contract):
    enabled = contract['providers']['sam3']
    if enabled:
        config = perception_entry.read_provider_config(path)
        perception_entry.verify_provider_config(config, contract)
    else:
        config = json.loads(Path(path).read_text())
        if (config.get('providers') != {'sam3': False, 'contact_graspnet': False}
                or config.get('provenance') != {'offline_skill_discovery': False}
                or config.get('grasp_calibration') is not None
                or any(key.endswith('_url') for key in config)):
            raise ValueError('disabled provider configuration differs from frozen contract')
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    root = Path('/work/aspire_validate')
    campaign = os.environ['ASPIRE_REPAIRED_CAMPAIGN']
    contract = repaired_subset_contract.load_contract(root, campaign)
    profile_evidence = repaired_subset_contract.verify_profile(root, contract)
    profile = (root / contract['source_profile']['relative_root']).resolve(strict=True)
    if Path(os.environ['ASPIRE_REPAIRED_PROFILE']).resolve(strict=True) != profile:
        raise ValueError('repaired profile environment differs from execution contract')
    manifest_path = root / 'full_results' / campaign / 'manifest.json'
    manifest, group = subset_entry.load_group_contract(manifest_path, args.task_name, args.seed)
    subset_entry.checked_layout_queue(os.environ.get('EVAL_LAYOUT_IDS'), group)
    subset_entry.checked_subset_count(group['native_target_episodes'],
        mode=os.environ.get('ASPIRE_FULL_MODE'), override=os.environ.get('EVAL_NUM'), group=group)
    output = Path(os.environ['ASPIRE_OUTPUT_DIR'])
    if output != root / 'full_results' / campaign / 'groups' / group['group_id']:
        raise ValueError('repaired subset output identity differs from contract')

    # Every import resolves within the same frozen profile, with no mutable
    # checkout fallback. The selected source paths are saved for final audit.
    perception_entry.activate_profile(profile)
    import full_entry
    import full_episode
    import aspire_glue
    import robodojo_adapter
    import receipt_executor
    import native_stdout
    import perception_bridge
    import cap_harness.api
    import cap_harness.contracts
    import cap_harness.runtime
    sources = {
        'qwen_full/repaired_subset_entry.py': __file__,
        'qwen_full/repaired_subset_inside.sh': os.environ['ASPIRE_BOOTSTRAP_INSIDE'],
        'qwen_full/repaired_subset_contract.py': repaired_subset_contract.__file__,
        'qwen_full/subset_entry.py': subset_entry.__file__,
        'qwen_full/perception_entry.py': perception_entry.__file__,
        'qwen_full/full_entry.py': full_entry.__file__,
        'qwen_full/full_episode.py': full_episode.__file__,
        'qwen_integration/aspire_glue.py': aspire_glue.__file__,
        'qwen_integration/robodojo_adapter.py': robodojo_adapter.__file__,
        'qwen_integration/receipt_executor.py': receipt_executor.__file__,
        'qwen_integration/native_stdout.py': native_stdout.__file__,
        'qwen_integration/perception_bridge.py': perception_bridge.__file__,
        'ASPIRE/src/cap_harness/api.py': cap_harness.api.__file__,
        'ASPIRE/src/cap_harness/contracts.py': cap_harness.contracts.__file__,
        'ASPIRE/src/cap_harness/runtime.py': cap_harness.runtime.__file__}
    bootstrap = repaired_subset_contract.verify_executed_sources(root, contract, sources)
    config_path = Path(os.environ['ASPIRE_PERCEPTION_CONFIG'])
    config = provider_configuration(config_path, contract)
    config_hash = repaired_subset_contract.sha_file(config_path)
    output.mkdir(parents=True, exist_ok=True)
    full_episode.write_json(output / 'repaired_subset_contract.json', {
        'protocol': contract['protocol'], 'campaign': campaign,
        'execution_contract_sha256': repaired_subset_contract.sha_file(
            root / 'full_results' / campaign / 'execution_manifest.json'),
        'dataset_manifest_sha256': repaired_subset_contract.sha_file(manifest_path),
        'group_id': group['group_id'], 'task': group['task'], 'eval_seed': group['eval_seed'],
        'step_limit': group['step_limit'], 'target_episodes': group['target_episodes'],
        'candidate_layout_ids': group['candidate_layout_ids'],
        'source_profile': contract['source_profile'], 'source_profile_verification': profile_evidence,
        'bootstrap_verification': bootstrap, 'providers': contract['providers'],
        'provider_config_sha256': config_hash})
    full_entry.effective_eval_count = lambda native_count, *, mode, override: subset_entry.checked_subset_count(
        native_count, mode=mode, override=override, group=group)
    enabled = contract['providers']['sam3']
    scope = ('Repaired ASPIRE API/program executor plus native X5/CuRobo bridge; '
             + ('SAM3 and Contact-GraspNet enabled; ' if enabled else 'learned perception disabled; ')
             + 'no VLA, offline skill discovery or cross-episode learning; custom 767-episode subset.')
    perception_entry.install_runtime_hooks(full_entry, full_episode,
        arm='perception' if enabled else 'baseline', group=group, config=config,
        config_sha256=config_hash,
        calibration_factory=perception_bridge.X5GraspCalibration.from_mapping if enabled else None,
        services_resident_for_both_arms=False, report_scope=scope)
    full_entry.main()


if __name__ == '__main__':
    main()
