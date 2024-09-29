GROMACS Molecular Dynamics (MD) Protocol
========================================

.. _md protocol api:

A Protocol for running MD simulation using GROMACS.


Protocol API Specification
--------------------------

.. module:: openfe_gromacs.protocols.gromacs_md

.. autosummary::
   :nosignatures:
   :toctree: generated/

   GromacsMDProtocol
   GromacsMDProtocolResult
   GromacsMDProtocolSettings
   GromacsMDSetupUnit
   GromacsMDRunUnit


Protocol Settings
-----------------

.. module:: openfe_gromacs.protocols.gromacs_md.md_settings

.. autopydantic_model:: GromacsMDProtocolSettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :exclude-members: get_defaults
   :member-order: bysource

