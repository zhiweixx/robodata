"""Bind the stopped A100 phase to its actual sources without relabeling scores."""
from pathlib import Path
import hashlib
import json

OLD_MANIFEST = '34270abf4a0ef8fc6272673d4d933ccbf316dc662724970a4af8508901faa5c3'
OLD_ALLOCATION = '28041155'
PHASE = 'maintenance/20261102_professor_api'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_transition(campaign, current):
    c = Path(campaign)
    phase = current.get('provider_transition', {})
    if (phase.get('directory') != PHASE or phase.get('prior_allocation_ids') != [OLD_ALLOCATION]
            or phase.get('prior_manifest_sha256') != OLD_MANIFEST):
        raise ValueError('provider transition must bind the single stopped A100 allocation')
    p = c / PHASE
    def read(name, expected):
        path = p / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError('provider transition archive changed: ' + name)
        return json.loads(path.read_text())
    old = read('manifest.json.before', OLD_MANIFEST)
    deployment = read('deployment.json.before', phase['prior_deployment_sha256'])
    if (deployment['manifest_sha256'] != OLD_MANIFEST
            or deployment['bootstrap_sha256'] != old['bootstrap_sha256']
            or old['model']['provider'] != 'local_vllm'
            or old['model']['served_alias'] != 'Qwen/Qwen3.8-27B'):
        raise ValueError('prior A100 provider/source identity mismatch')
    for name, expected in old['bootstrap_sha256'].items():
        if digest(p/'bootstrap.before'/name) != expected:
            raise ValueError('archived A100 bootstrap changed: ' + name)
    allowed = {'model', 'bootstrap_relative_root', 'bootstrap_sha256',
        'deployment_preflight_file', 'deployment_preflight_sha256',
        'provider_transition', 'sampling_difference', 'pre_request_bootstrap_repairs'}
    for key in set(old) | set(current):
        if key not in allowed and old.get(key) != current.get(key):
            raise ValueError('provider switch changed preserved experiment field: ' + key)
    report = read('terminal_a100_audit.json', phase['terminal_audit_sha256'])
    if (report['issues'] or report['evidence_audited'] is not True
            or report['scored_episodes'] != 11 or report['verified_episode_reports'] != 11
            or report['nonempty_video_count'] != 33):
        raise ValueError('prior native results did not pass the terminal audit')
    allocation = json.loads((c/'allocations'/OLD_ALLOCATION/'allocation.json').read_text())
    if (allocation.get('manifest_sha256') != OLD_MANIFEST
            or not allocation.get('finished_utc') or allocation.get('children_stopped') is not True
            or allocation.get('cleanup_errors')):
        raise ValueError('prior A100 evaluation has not stopped cleanly')
    restore = read('provider_restore_receipt.json', phase['restore_receipt_sha256'])
    if (restore.get('inference', {}).get('valid_complete_exact_model_response') is not True
            or restore['inference'].get('response_model') != 'qwen38-27b'
            or restore.get('network_cleanup', {}).get('network_cleanup_verified') is not True):
        raise ValueError('restored professor API receipt is invalid')
    bindings = {OLD_ALLOCATION: {'manifest': old, 'manifest_sha256': OLD_MANIFEST}}
    for repair in current.get('pre_request_bootstrap_repairs', []):
        if (repair.get('directory') != PHASE+'/launcher_model_alias'
                or repair.get('allocation') != '28041348'
                or repair.get('manifest_sha256') != '695c6905f3224164c656928ff0d3b1c1b7c3e9d23ff210bff5139e448147a2a9'):
            raise ValueError('unexpected pre-request launcher repair')
        archive = c/repair['directory']
        if digest(archive/'manifest.json.before') != repair['manifest_sha256']:
            raise ValueError('launcher repair prior manifest changed')
        previous = json.loads((archive/'manifest.json.before').read_text())
        for name, expected in previous['bootstrap_sha256'].items():
            if digest(archive/'bootstrap.before'/name) != expected:
                raise ValueError('launcher repair archived source changed: '+name)
        for key in set(previous) | set(current):
            if key not in {'bootstrap_sha256','deployment_preflight_file',
                    'deployment_preflight_sha256','pre_request_bootstrap_repairs'} and previous.get(key) != current.get(key):
                raise ValueError('launcher repair changed experiment field: '+key)
        directory = c/'allocations'/repair['allocation']
        receipt = json.loads((directory/'allocation.json').read_text())
        if (receipt.get('manifest_sha256') != repair['manifest_sha256']
                or not receipt.get('finished_utc') or receipt.get('children_stopped') is not True
                or receipt.get('cleanup_errors')
                or receipt.get('network_cleanup',{}).get('network_cleanup_verified') is not True
                or list(directory.glob('workers/*/*/raw_inference/*.sse'))):
            raise ValueError('launcher repair requires clean stopped pre-request allocation')
        for usage in directory.glob('workers/*/*/usage.jsonl'):
            if any(json.loads(line).get('event')=='generation_started' for line in usage.read_text().splitlines()):
                raise ValueError('cannot silently retry a model request after launcher failure')
        bindings[repair['allocation']] = {'manifest':previous,'manifest_sha256':repair['manifest_sha256']}
    return {'verified': True, 'billing_mode': 'user_managed',
        'phases': [{'directory': PHASE, 'prior_allocation_ids': [OLD_ALLOCATION],
            'prior_provider': 'local_vllm', 'current_provider': 'professor_api',
            'remote_checkpoint_verified': False, 'preserved_native_scores': 11}],
        'allocations': bindings}
