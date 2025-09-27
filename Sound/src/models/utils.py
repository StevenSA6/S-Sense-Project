def infer_in_channels(cfg) -> int:
  # base mel
  base = 1
  if cfg.features.deltas.enabled:
    base += cfg.features.deltas.order  # +1 for Δ, +2 for Δ,ΔΔ

  aux = 0
  if cfg.features.aux_channels.enabled:
    ac = cfg.features.aux_channels
    aux += int(ac.spectral_flux)
    aux += int(ac.centroid)
    aux += int(ac.zcr)
    aux += int(ac.percussive_mask_mean)
    # make sure preprocess.envelope_aux.enabled matches this
    aux += int(ac.envelope_rms)
  return base + aux
