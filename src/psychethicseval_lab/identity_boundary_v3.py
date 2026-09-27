"""Final v13 permissions; model assertions never grant operational authority."""
from .identity_classifier_v13 import validate_v13_decision_record


def validate_decision_record_v3(record):
    classifier = record.get('classifier_id') if isinstance(record, dict) else None
    if classifier != 'identity-classifier-v13':
        return False, ['v13_required']
    return validate_v13_decision_record(record)


def permissions_for_v3(record):
    if not validate_decision_record_v3(record)[0]:
        raise ValueError('v3_requires_valid_v13')
    permissions = ['universal', 'domain_safe']
    if record['permission_basis'] in ('source_bound', 'trusted_metadata'):
        permissions.append('framing:' + record['role'])
    if record['permission_basis'] == 'trusted_metadata':
        permissions.append('role_specific:' + record['role'])
    return tuple(permissions)
