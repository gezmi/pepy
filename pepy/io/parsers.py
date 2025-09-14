"""
Format-specific parsers for confidence metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


class ConfidenceMetricsParser:
	"""
	Parse confidence metrics into standardized format.
	"""

	@staticmethod
	def parse_confidence_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Parse raw confidence data into standardized metrics.

		Args:
			raw_data: Raw confidence data from ConfidenceFileLoader

		Returns:
			Dict with standardized confidence metrics
		"""
		# Convert PAE matrix to DataFrame if it exists
		pae_matrix = None
		if raw_data.get('pae_matrix') is not None:
			pae_array = np.array(raw_data['pae_matrix'])
			if pae_array.ndim >= 2:
				# Take first matrix if multiple models
				if pae_array.ndim == 3:
					pae_array = pae_array[0]
				pae_matrix = pd.DataFrame(pae_array)

		# Extract scalar metrics
		iptm = ConfidenceMetricsParser._to_float(raw_data.get('iptm'))
		ptm = ConfidenceMetricsParser._to_float(raw_data.get('ptm'))

		# Calculate combined confidence if both metrics available
		confidence = None
		if iptm is not None and ptm is not None:
			confidence = round(0.8 * iptm + 0.2 * ptm, 2)

		# Parse pLDDT data
		plddt_data = raw_data.get('plddt')

		return {
			'pae_matrix': pae_matrix,
			'iptm': iptm,
			'ptm': ptm,
			'confidence': confidence,
			'plddt': plddt_data,
			'has_confidence_data': any([
				pae_matrix is not None,
				iptm is not None,
				ptm is not None,
				plddt_data is not None
			])
		}

	@staticmethod
	def _to_float(value: Any) -> Optional[float]:
		"""Convert value to float, handling various formats."""
		if value is None:
			return None

		try:
			# Handle numpy arrays
			if hasattr(value, 'item'):
				return float(value.item())
			# Handle lists
			elif isinstance(value, (list, tuple)) and len(value) > 0:
				return float(value[0])
			# Handle scalars
			else:
				return float(value)
		except (ValueError, TypeError, IndexError):
			return None