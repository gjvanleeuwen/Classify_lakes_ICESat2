# view a short list of order IDs
region.granules.orderIDs


region_a.visualize_spatial_extent()


cyclemap, rgtmap = region.visualize_elevation()

# atl03_pad, band_tol, min_full_sat, min_near_sat, min_sat_h, min_scan_s, ph_sat_flag, ph_sat_lb, ph_sat_ub,
# podppd_pad, scan_settle_s, det_ab_flag, ds_gt, ds_stat, hvpc_ab_flag, laser_12_flag, lrs_ab_flag, pdu_ab_flag,
# ph_uncorrelated_error, rx_bckgrd_sensitivity, rx_return_sensitivity, spd_ab_flag, tams_ab_flag, tx_pulse_distribution,
# tx_pulse_energy, tx_pulse_skew_est, tx_pulse_thresh_lower, tx_pulse_thresh_upper, tx_pulse_width_lower, tx_pulse_width_upper,
# atlas_sdp_gps_epoch, cal42_product, dead_time, sigma, side, temperature, cal34_product, rad_corr, strength, width, ds_channel,
# cal19_product, ffb_corr, bin_width, cal20_product, hist, total_events, hist_x, laser, mode, num_bins, return_source, control,
# data_end_utc, data_start_utc, end_cycle, end_delta_time, end_geoseg, end_gpssow, end_gpsweek, end_orbit, end_region, end_rgt,
# granule_end_utc, granule_start_utc, addpad_flag, alpha_inc, alpha_max, delta_t_gap_min, delta_t_lin_fit, delta_t_max,
# delta_t_min, delta_z_bg, delta_zmax2, delta_zmin, e_a, e_linfit_edit, e_linfit_slant, e_m, e_m_mult, htspanmin, lslant_flag,
# min_fit_time_fact, n_delta_z1, n_delta_z2, nbin_min, nphot_min, nslw, nslw_v, out_edit_flag, pc_bckgrd_flag, r, r2, sig_find_t_inc,
# snrlow, snrmed, t_gap_big, release, start_cycle, start_delta_time, start_geoseg, start_gpssow, start_gpsweek, start_orbit, start_region,
# start_rgt, min_tep_ph, min_tep_secs, n_tep_bins, tep_bin_size, tep_gap_size, tep_normalize, tep_peak_bins, tep_prim_window, tep_range_prim,
# tep_rm_noise, tep_sec_window, tep_start_x, tep_valid_spot, version, reference_tep_flag, tep_bckgrd, tep_duration, tep_hist, tep_hist_sum,
# tep_hist_time, tep_tod, ds_surf_type, ds_xyz, bckgrd_counts, bckgrd_counts_reduced, bckgrd_hist_top, bckgrd_int_height, bckgrd_int_height_reduced,
# bckgrd_rate, delta_time, pce_mframe_cnt, tlm_height_band1, tlm_height_band2, tlm_top_band1, tlm_top_band2, altitude_sc, bounce_time_offset,
# full_sat_fract, near_sat_fract, neutat_delay_derivative, neutat_delay_total, neutat_ht, ph_index_beg, pitch, podppd_flag, range_bias_corr,
# ref_azimuth, ref_elev, reference_photon_index, reference_photon_lat, reference_photon_lon, roll, segment_dist_x, segment_id, segment_length,
# segment_ph_cnt, sigma_across, sigma_along, sigma_h, sigma_lat, sigma_lon, solar_azimuth, solar_elevation, surf_type, velocity_sc, yaw, dac,
# dem_flag, dem_h, geoid, geoid_free2mean, tide_earth, tide_earth_free2mean, tide_equilibrium, tide_load, tide_oc_pole, tide_ocean, tide_pole,
# dist_ph_across, dist_ph_along, h_ph, lat_ph, lon_ph, ph_id_channel, ph_id_count, ph_id_pulse, quality_ph, signal_conf_ph, bckgrd_mean, bckgrd_sigma,
# t_pc_delta, z_pc_delta, crossing_time, cycle_number, lan, orbit_number, rgt, sc_orient, sc_orient_time, qa_perc_signal_conf_ph_high,
# qa_perc_signal_conf_ph_low, qa_perc_signal_conf_ph_med, qa_perc_surf_type, qa_total_signal_conf_ph_high, qa_total_signal_conf_ph_low,
# qa_total_signal_conf_ph_med, qa_granule_fail_reason, qa_granule_pass_fail



# 'ancillary_data/altimetry/atl03_pad', 'ancillary_data/altimetry/band_tol', 'ancillary_data/altimetry/min_full_sat',
    # 'ancillary_data/altimetry/min_near_sat', 'ancillary_data/altimetry/min_sat_h', 'ancillary_data/altimetry/min_scan_s',
    # 'ancillary_data/altimetry/ph_sat_flag', 'ancillary_data/altimetry/ph_sat_lb', 'ancillary_data/altimetry/ph_sat_ub',
    # 'ancillary_data/altimetry/podppd_pad', 'ancillary_data/altimetry/scan_settle_s', 'ancillary_data/atlas_engineering/det_ab_flag',
    # 'ancillary_data/atlas_engineering/ds_gt', 'ancillary_data/atlas_engineering/ds_stat', 'ancillary_data/atlas_engineering/hvpc_ab_flag',
    # 'ancillary_data/atlas_engineering/laser_12_flag', 'ancillary_data/atlas_engineering/lrs_ab_flag', 'ancillary_data/atlas_engineering/pdu_ab_flag',
    # 'ancillary_data/atlas_engineering/ph_uncorrelated_error', 'ancillary_data/atlas_engineering/receiver/rx_bckgrd_sensitivity',
    # 'ancillary_data/atlas_engineering/receiver/rx_return_sensitivity', 'ancillary_data/atlas_engineering/spd_ab_flag',
    # 'ancillary_data/atlas_engineering/tams_ab_flag', 'ancillary_data/atlas_engineering/transmit/tx_pulse_distribution',
    # 'ancillary_data/atlas_engineering/transmit/tx_pulse_energy', 'ancillary_data/atlas_engineering/transmit/tx_pulse_skew_est',
    # 'ancillary_data/atlas_engineering/transmit/tx_pulse_thresh_lower', 'ancillary_data/atlas_engineering/transmit/tx_pulse_thresh_upper',
    # 'ancillary_data/atlas_engineering/transmit/tx_pulse_width_lower', 'ancillary_data/atlas_engineering/transmit/tx_pulse_width_upper',
    # 'ancillary_data/atlas_sdp_gps_epoch', 'ancillary_data/calibrations/dead_time/cal42_product', 'ancillary_data/calibrations/dead_time/gt1l/dead_time',
    # 'ancillary_data/calibrations/dead_time/gt1l/sigma', 'ancillary_data/calibrations/dead_time/gt1r/dead_time', 'ancillary_data/calibrations/dead_time/gt1r/sigma',
    # 'ancillary_data/calibrations/dead_time/gt2l/dead_time', 'ancillary_data/calibrations/dead_time/gt2l/sigma',
    # 'ancillary_data/calibrations/dead_time/gt2r/dead_time', 'ancillary_data/calibrations/dead_time/gt2r/sigma',
    # 'ancillary_data/calibrations/dead_time/gt3l/dead_time', 'ancillary_data/calibrations/dead_time/gt3l/sigma',
    # 'ancillary_data/calibrations/dead_time/gt3r/dead_time', 'ancillary_data/calibrations/dead_time/gt3r/sigma',
    # 'ancillary_data/calibrations/dead_time/side', 'ancillary_data/calibrations/dead_time/temperature',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/cal34_product',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt1l/dead_time',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt1l/rad_corr',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt1l/strength',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt1l/width', '
    # ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt1r/dead_time',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt1r/rad_corr',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt1r/strength',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt1r/width',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt2l/dead_time', 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt2l/rad_corr',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt2l/strength', 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt2l/width',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt2r/dead_time', 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt2r/rad_corr',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt2r/strength', 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt2r/width',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt3l/dead_time', 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt3l/rad_corr',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt3l/strength', 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt3l/width',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt3r/dead_time', 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt3r/rad_corr',
    # 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt3r/strength', 'ancillary_data/calibrations/dead_time_radiometric_signal_loss/gt3r/width',
    # 'ancillary_data/calibrations/ds_channel', 'ancillary_data/calibrations/first_photon_bias/cal19_product', 'ancillary_data/calibrations/first_photon_bias/gt1l/dead_time',
    # 'ancillary_data/calibrations/first_photon_bias/gt1l/ffb_corr', 'ancillary_data/calibrations/first_photon_bias/gt1l/strength',
    # 'ancillary_data/calibrations/first_photon_bias/gt1l/width', 'ancillary_data/calibrations/first_photon_bias/gt1r/dead_time',
    # 'ancillary_data/calibrations/first_photon_bias/gt1r/ffb_corr', 'ancillary_data/calibrations/first_photon_bias/gt1r/strength',
    # 'ancillary_data/calibrations/first_photon_bias/gt1r/width', 'ancillary_data/calibrations/first_photon_bias/gt2l/dead_time',
    # 'ancillary_data/calibrations/first_photon_bias/gt2l/ffb_corr', 'ancillary_data/calibrations/first_photon_bias/gt2l/strength',
    # 'ancillary_data/calibrations/first_photon_bias/gt2l/width', 'ancillary_data/calibrations/first_photon_bias/gt2r/dead_time',
    # 'ancillary_data/calibrations/first_photon_bias/gt2r/ffb_corr', 'ancillary_data/calibrations/first_photon_bias/gt2r/strength',
    # 'ancillary_data/calibrations/first_photon_bias/gt2r/width', 'ancillary_data/calibrations/first_photon_bias/gt3l/dead_time',
    # 'ancillary_data/calibrations/first_photon_bias/gt3l/ffb_corr', 'ancillary_data/calibrations/first_photon_bias/gt3l/strength',
    # 'ancillary_data/calibrations/first_photon_bias/gt3l/width', 'ancillary_data/calibrations/first_photon_bias/gt3r/dead_time',
    # 'ancillary_data/calibrations/first_photon_bias/gt3r/ffb_corr', 'ancillary_data/calibrations/first_photon_bias/gt3r/strength',
    # 'ancillary_data/calibrations/first_photon_bias/gt3r/width', 'ancillary_data/calibrations/low_link_impulse_response/bin_width',
    # 'ancillary_data/calibrations/low_link_impulse_response/cal20_product', 'ancillary_data/calibrations/low_link_impulse_response/gt1l/hist',
    # 'ancillary_data/calibrations/low_link_impulse_response/gt1l/total_events', 'ancillary_data/calibrations/low_link_impulse_response/gt1r/hist',
    # 'ancillary_data/calibrations/low_link_impulse_response/gt1r/total_events', 'ancillary_data/calibrations/low_link_impulse_response/gt2l/hist',
    # 'ancillary_data/calibrations/low_link_impulse_response/gt2l/total_events', 'ancillary_data/calibrations/low_link_impulse_response/gt2r/hist',
    # 'ancillary_data/calibrations/low_link_impulse_response/gt2r/total_events', 'ancillary_data/calibrations/low_link_impulse_response/gt3l/hist',
    # 'ancillary_data/calibrations/low_link_impulse_response/gt3l/total_events', 'ancillary_data/calibrations/low_link_impulse_response/gt3r/hist',
    # 'ancillary_data/calibrations/low_link_impulse_response/gt3r/total_events', 'ancillary_data/calibrations/low_link_impulse_response/hist_x',
    # 'ancillary_data/calibrations/low_link_impulse_response/laser', 'ancillary_data/calibrations/low_link_impulse_response/mode',
    # 'ancillary_data/calibrations/low_link_impulse_response/num_bins', 'ancillary_data/calibrations/low_link_impulse_response/return_source',
    # 'ancillary_data/calibrations/low_link_impulse_response/side', 'ancillary_data/calibrations/low_link_impulse_response/temperature',
    # 'ancillary_data/control', 'ancillary_data/data_end_utc', 'ancillary_data/data_start_utc', 'ancillary_data/end_cycle', 'ancillary_data/end_delta_time',
    # 'ancillary_data/end_geoseg', 'ancillary_data/end_gpssow', 'ancillary_data/end_gpsweek', 'ancillary_data/end_orbit', 'ancillary_data/end_region',
    # 'ancillary_data/end_rgt', 'ancillary_data/granule_end_utc', 'ancillary_data/granule_start_utc', 'ancillary_data/gt1l/signal_find_input/addpad_flag',
    # 'ancillary_data/gt1l/signal_find_input/alpha_inc', 'ancillary_data/gt1l/signal_find_input/alpha_max', 'ancillary_data/gt1l/signal_find_input/delta_t_gap_min',
    # 'ancillary_data/gt1l/signal_find_input/delta_t_lin_fit', 'ancillary_data/gt1l/signal_find_input/delta_t_max', 'ancillary_data/gt1l/signal_find_input/delta_t_min',
    # 'ancillary_data/gt1l/signal_find_input/delta_z_bg', 'ancillary_data/gt1l/signal_find_input/delta_zmax2', 'ancillary_data/gt1l/signal_find_input/delta_zmin',
    # 'ancillary_data/gt1l/signal_find_input/e_a', 'ancillary_data/gt1l/signal_find_input/e_linfit_edit', 'ancillary_data/gt1l/signal_find_input/e_linfit_slant',
    # 'ancillary_data/gt1l/signal_find_input/e_m', 'ancillary_data/gt1l/signal_find_input/e_m_mult', 'ancillary_data/gt1l/signal_find_input/htspanmin',
    # 'ancillary_data/gt1l/signal_find_input/lslant_flag', 'ancillary_data/gt1l/signal_find_input/min_fit_time_fact',
    # 'ancillary_data/gt1l/signal_find_input/n_delta_z1', 'ancillary_data/gt1l/signal_find_input/n_delta_z2', 'ancillary_data/gt1l/signal_find_input/nbin_min',
    # 'ancillary_data/gt1l/signal_find_input/nphot_min', 'ancillary_data/gt1l/signal_find_input/nslw', 'ancillary_data/gt1l/signal_find_input/nslw_v',
    # 'ancillary_data/gt1l/signal_find_input/out_edit_flag', 'ancillary_data/gt1l/signal_find_input/pc_bckgrd_flag', 'ancillary_data/gt1l/signal_find_input/r',
    # 'ancillary_data/gt1l/signal_find_input/r2', 'ancillary_data/gt1l/signal_find_input/sig_find_t_inc', 'ancillary_data/gt1l/signal_find_input/snrlow',
    # 'ancillary_data/gt1l/signal_find_input/snrmed', 'ancillary_data/gt1l/signal_find_input/t_gap_big', 'ancillary_data/gt1r/signal_find_input/addpad_flag',
    # 'ancillary_data/gt1r/signal_find_input/alpha_inc', 'ancillary_data/gt1r/signal_find_input/alpha_max', 'ancillary_data/gt1r/signal_find_input/delta_t_gap_min',
    # 'ancillary_data/gt1r/signal_find_input/delta_t_lin_fit', 'ancillary_data/gt1r/signal_find_input/delta_t_max', 'ancillary_data/gt1r/signal_find_input/delta_t_min',
    # 'ancillary_data/gt1r/signal_find_input/delta_z_bg', 'ancillary_data/gt1r/signal_find_input/delta_zmax2',
    # 'ancillary_data/gt1r/signal_find_input/delta_zmin', 'ancillary_data/gt1r/signal_find_input/e_a',
    # 'ancillary_data/gt1r/signal_find_input/e_linfit_edit', 'ancillary_data/gt1r/signal_find_input/e_linfit_slant',
    # 'ancillary_data/gt1r/signal_find_input/e_m', 'ancillary_data/gt1r/signal_find_input/e_m_mult',
    # 'ancillary_data/gt1r/signal_find_input/htspanmin', 'ancillary_data/gt1r/signal_find_input/lslant_flag',
    # 'ancillary_data/gt1r/signal_find_input/min_fit_time_fact', 'ancillary_data/gt1r/signal_find_input/n_delta_z1',
    # 'ancillary_data/gt1r/signal_find_input/n_delta_z2', 'ancillary_data/gt1r/signal_find_input/nbin_min',
    # 'ancillary_data/gt1r/signal_find_input/nphot_min', 'ancillary_data/gt1r/signal_find_input/nslw',
    # 'ancillary_data/gt1r/signal_find_input/nslw_v', 'ancillary_data/gt1r/signal_find_input/out_edit_flag',
    # 'ancillary_data/gt1r/signal_find_input/pc_bckgrd_flag', 'ancillary_data/gt1r/signal_find_input/r',
    # 'ancillary_data/gt1r/signal_find_input/r2', 'ancillary_data/gt1r/signal_find_input/sig_find_t_inc',
    # 'ancillary_data/gt1r/signal_find_input/snrlow', 'ancillary_data/gt1r/signal_find_input/snrmed',
    # 'ancillary_data/gt1r/signal_find_input/t_gap_big', 'ancillary_data/gt2l/signal_find_input/addpad_flag',
    # 'ancillary_data/gt2l/signal_find_input/alpha_inc', 'ancillary_data/gt2l/signal_find_input/alpha_max',
    # 'ancillary_data/gt2l/signal_find_input/delta_t_gap_min', 'ancillary_data/gt2l/signal_find_input/delta_t_lin_fit',
    # 'ancillary_data/gt2l/signal_find_input/delta_t_max', 'ancillary_data/gt2l/signal_find_input/delta_t_min',
    # 'ancillary_data/gt2l/signal_find_input/delta_z_bg', 'ancillary_data/gt2l/signal_find_input/delta_zmax2',
    # 'ancillary_data/gt2l/signal_find_input/delta_zmin', 'ancillary_data/gt2l/signal_find_input/e_a',
    # 'ancillary_data/gt2l/signal_find_input/e_linfit_edit', 'ancillary_data/gt2l/signal_find_input/e_linfit_slant',
    # 'ancillary_data/gt2l/signal_find_input/e_m', 'ancillary_data/gt2l/signal_find_input/e_m_mult', 'ancillary_data/gt2l/signal_find_input/htspanmin',
    # 'ancillary_data/gt2l/signal_find_input/lslant_flag', 'ancillary_data/gt2l/signal_find_input/min_fit_time_fact',
    # 'ancillary_data/gt2l/signal_find_input/n_delta_z1', 'ancillary_data/gt2l/signal_find_input/n_delta_z2',
    # 'ancillary_data/gt2l/signal_find_input/nbin_min', 'ancillary_data/gt2l/signal_find_input/nphot_min',
    # 'ancillary_data/gt2l/signal_find_input/nslw', 'ancillary_data/gt2l/signal_find_input/nslw_v', 'ancillary_data/gt2l/signal_find_input/out_edit_flag',
    # 'ancillary_data/gt2l/signal_find_input/pc_bckgrd_flag', 'ancillary_data/gt2l/signal_find_input/r', 'ancillary_data/gt2l/signal_find_input/r2',
    # 'ancillary_data/gt2l/signal_find_input/sig_find_t_inc', 'ancillary_data/gt2l/signal_find_input/snrlow', 'ancillary_data/gt2l/signal_find_input/snrmed',
    # 'ancillary_data/gt2l/signal_find_input/t_gap_big', 'ancillary_data/gt2r/signal_find_input/addpad_flag', 'ancillary_data/gt2r/signal_find_input/alpha_inc',
    # 'ancillary_data/gt2r/signal_find_input/alpha_max', 'ancillary_data/gt2r/signal_find_input/delta_t_gap_min', 'ancillary_data/gt2r/signal_find_input/delta_t_lin_fit',
    # 'ancillary_data/gt2r/signal_find_input/delta_t_max', 'ancillary_data/gt2r/signal_find_input/delta_t_min', 'ancillary_data/gt2r/signal_find_input/delta_z_bg',
    # 'ancillary_data/gt2r/signal_find_input/delta_zmax2', 'ancillary_data/gt2r/signal_find_input/delta_zmin', 'ancillary_data/gt2r/signal_find_input/e_a',
    # 'ancillary_data/gt2r/signal_find_input/e_linfit_edit', 'ancillary_data/gt2r/signal_find_input/e_linfit_slant', 'ancillary_data/gt2r/signal_find_input/e_m',
    # 'ancillary_data/gt2r/signal_find_input/e_m_mult', 'ancillary_data/gt2r/signal_find_input/htspanmin', 'ancillary_data/gt2r/signal_find_input/lslant_flag',
    # 'ancillary_data/gt2r/signal_find_input/min_fit_time_fact', 'ancillary_data/gt2r/signal_find_input/n_delta_z1', 'ancillary_data/gt2r/signal_find_input/n_delta_z2',
    # 'ancillary_data/gt2r/signal_find_input/nbin_min', 'ancillary_data/gt2r/signal_find_input/nphot_min', 'ancillary_data/gt2r/signal_find_input/nslw',
    # 'ancillary_data/gt2r/signal_find_input/nslw_v', 'ancillary_data/gt2r/signal_find_input/out_edit_flag', 'ancillary_data/gt2r/signal_find_input/pc_bckgrd_flag',
    # 'ancillary_data/gt2r/signal_find_input/r', 'ancillary_data/gt2r/signal_find_input/r2', 'ancillary_data/gt2r/signal_find_input/sig_find_t_inc',
    # 'ancillary_data/gt2r/signal_find_input/snrlow', 'ancillary_data/gt2r/signal_find_input/snrmed', 'ancillary_data/gt2r/signal_find_input/t_gap_big',
    # 'ancillary_data/gt3l/signal_find_input/addpad_flag', 'ancillary_data/gt3l/signal_find_input/alpha_inc', 'ancillary_data/gt3l/signal_find_input/alpha_max',
    # 'ancillary_data/gt3l/signal_find_input/delta_t_gap_min', 'ancillary_data/gt3l/signal_find_input/delta_t_lin_fit', 'ancillary_data/gt3l/signal_find_input/delta_t_max',
    # 'ancillary_data/gt3l/signal_find_input/delta_t_min', 'ancillary_data/gt3l/signal_find_input/delta_z_bg', 'ancillary_data/gt3l/signal_find_input/delta_zmax2',
    # 'ancillary_data/gt3l/signal_find_input/delta_zmin', 'ancillary_data/gt3l/signal_find_input/e_a', 'ancillary_data/gt3l/signal_find_input/e_linfit_edit',
    # 'ancillary_data/gt3l/signal_find_input/e_linfit_slant', 'ancillary_data/gt3l/signal_find_input/e_m', 'ancillary_data/gt3l/signal_find_input/e_m_mult',
    # 'ancillary_data/gt3l/signal_find_input/htspanmin', 'ancillary_data/gt3l/signal_find_input/lslant_flag', 'ancillary_data/gt3l/signal_find_input/min_fit_time_fact',
    # 'ancillary_data/gt3l/signal_find_input/n_delta_z1', 'ancillary_data/gt3l/signal_find_input/n_delta_z2', 'ancillary_data/gt3l/signal_find_input/nbin_min',
    # 'ancillary_data/gt3l/signal_find_input/nphot_min', 'ancillary_data/gt3l/signal_find_input/nslw', 'ancillary_data/gt3l/signal_find_input/nslw_v',
    # 'ancillary_data/gt3l/signal_find_input/out_edit_flag', 'ancillary_data/gt3l/signal_find_input/pc_bckgrd_flag', 'ancillary_data/gt3l/signal_find_input/r',
    # 'ancillary_data/gt3l/signal_find_input/r2', 'ancillary_data/gt3l/signal_find_input/sig_find_t_inc', 'ancillary_data/gt3l/signal_find_input/snrlow',
    # 'ancillary_data/gt3l/signal_find_input/snrmed', 'ancillary_data/gt3l/signal_find_input/t_gap_big', 'ancillary_data/gt3r/signal_find_input/addpad_flag',
    # 'ancillary_data/gt3r/signal_find_input/alpha_inc', 'ancillary_data/gt3r/signal_find_input/alpha_max', 'ancillary_data/gt3r/signal_find_input/delta_t_gap_min',
    # 'ancillary_data/gt3r/signal_find_input/delta_t_lin_fit', 'ancillary_data/gt3r/signal_find_input/delta_t_max', 'ancillary_data/gt3r/signal_find_input/delta_t_min',
    # 'ancillary_data/gt3r/signal_find_input/delta_z_bg', 'ancillary_data/gt3r/signal_find_input/delta_zmax2', 'ancillary_data/gt3r/signal_find_input/delta_zmin',
    # 'ancillary_data/gt3r/signal_find_input/e_a', 'ancillary_data/gt3r/signal_find_input/e_linfit_edit', 'ancillary_data/gt3r/signal_find_input/e_linfit_slant',
    # 'ancillary_data/gt3r/signal_find_input/e_m', 'ancillary_data/gt3r/signal_find_input/e_m_mult', 'ancillary_data/gt3r/signal_find_input/htspanmin',
    # 'ancillary_data/gt3r/signal_find_input/lslant_flag', 'ancillary_data/gt3r/signal_find_input/min_fit_time_fact', 'ancillary_data/gt3r/signal_find_input/n_delta_z1',
    # 'ancillary_data/gt3r/signal_find_input/n_delta_z2', 'ancillary_data/gt3r/signal_find_input/nbin_min', 'ancillary_data/gt3r/signal_find_input/nphot_min',
    # 'ancillary_data/gt3r/signal_find_input/nslw', 'ancillary_data/gt3r/signal_find_input/nslw_v', 'ancillary_data/gt3r/signal_find_input/out_edit_flag',
    # 'ancillary_data/gt3r/signal_find_input/pc_bckgrd_flag', 'ancillary_data/gt3r/signal_find_input/r', 'ancillary_data/gt3r/signal_find_input/r2',
    # 'ancillary_data/gt3r/signal_find_input/sig_find_t_inc', 'ancillary_data/gt3r/signal_find_input/snrlow', 'ancillary_data/gt3r/signal_find_input/snrmed',
    # 'ancillary_data/gt3r/signal_find_input/t_gap_big', 'ancillary_data/release', 'ancillary_data/start_cycle', 'ancillary_data/start_delta_time',
    # 'ancillary_data/start_geoseg', 'ancillary_data/start_gpssow', 'ancillary_data/start_gpsweek', 'ancillary_data/start_orbit', 'ancillary_data/start_region',
    # 'ancillary_data/start_rgt', 'ancillary_data/tep/ds_gt', 'ancillary_data/tep/min_tep_ph', 'ancillary_data/tep/min_tep_secs', 'ancillary_data/tep/n_tep_bins', 'ancillary_data/tep/tep_bin_size', 'ancillary_data/tep/tep_gap_size', 'ancillary_data/tep/tep_normalize', 'ancillary_data/tep/tep_peak_bins', 'ancillary_data/tep/tep_prim_window', 'ancillary_data/tep/tep_range_prim', 'ancillary_data/tep/tep_rm_noise', 'ancillary_data/tep/tep_sec_window', 'ancillary_data/tep/tep_start_x', 'ancillary_data/tep/tep_valid_spot', 'ancillary_data/version', 'atlas_impulse_response/pce1_spot1/tep_histogram/reference_tep_flag', 'atlas_impulse_response/pce1_spot1/tep_histogram/tep_bckgrd', 'atlas_impulse_response/pce1_spot1/tep_histogram/tep_duration', 'atlas_impulse_response/pce1_spot1/tep_histogram/tep_hist', 'atlas_impulse_response/pce1_spot1/tep_histogram/tep_hist_sum', 'atlas_impulse_response/pce1_spot1/tep_histogram/tep_hist_time', 'atlas_impulse_response/pce1_spot1/tep_histogram/tep_tod', 'atlas_impulse_response/pce2_spot3/tep_histogram/reference_tep_flag', 'atlas_impulse_response/pce2_spot3/tep_histogram/tep_bckgrd', 'atlas_impulse_response/pce2_spot3/tep_histogram/tep_duration', 'atlas_impulse_response/pce2_spot3/tep_histogram/tep_hist', 'atlas_impulse_response/pce2_spot3/tep_histogram/tep_hist_sum', 'atlas_impulse_response/pce2_spot3/tep_histogram/tep_hist_time', 'atlas_impulse_response/pce2_spot3/tep_histogram/tep_tod', 'ds_surf_type', 'ds_xyz', 'gt1l/bckgrd_atlas/bckgrd_counts', 'gt1l/bckgrd_atlas/bckgrd_counts_reduced', 'gt1l/bckgrd_atlas/bckgrd_hist_top', 'gt1l/bckgrd_atlas/bckgrd_int_height', 'gt1l/bckgrd_atlas/bckgrd_int_height_reduced', 'gt1l/bckgrd_atlas/bckgrd_rate', 'gt1l/bckgrd_atlas/delta_time', 'gt1l/bckgrd_atlas/pce_mframe_cnt', 'gt1l/bckgrd_atlas/tlm_height_band1', 'gt1l/bckgrd_atlas/tlm_height_band2', 'gt1l/bckgrd_atlas/tlm_top_band1', 'gt1l/bckgrd_atlas/tlm_top_band2', 'gt1l/geolocation/altitude_sc', 'gt1l/geolocation/bounce_time_offset', 'gt1l/geolocation/delta_time', 'gt1l/geolocation/full_sat_fract', 'gt1l/geolocation/near_sat_fract', 'gt1l/geolocation/neutat_delay_derivative', 'gt1l/geolocation/neutat_delay_total', 'gt1l/geolocation/neutat_ht', 'gt1l/geolocation/ph_index_beg', 'gt1l/geolocation/pitch', 'gt1l/geolocation/podppd_flag', 'gt1l/geolocation/range_bias_corr', 'gt1l/geolocation/ref_azimuth', 'gt1l/geolocation/ref_elev', 'gt1l/geolocation/reference_photon_index', 'gt1l/geolocation/reference_photon_lat', 'gt1l/geolocation/reference_photon_lon', 'gt1l/geolocation/roll', 'gt1l/geolocation/segment_dist_x', 'gt1l/geolocation/segment_id', 'gt1l/geolocation/segment_length', 'gt1l/geolocation/segment_ph_cnt', 'gt1l/geolocation/sigma_across', 'gt1l/geolocation/sigma_along', 'gt1l/geolocation/sigma_h', 'gt1l/geolocation/sigma_lat', 'gt1l/geolocation/sigma_lon', 'gt1l/geolocation/solar_azimuth', 'gt1l/geolocation/solar_elevation', 'gt1l/geolocation/surf_type', 'gt1l/geolocation/tx_pulse_energy', 'gt1l/geolocation/tx_pulse_skew_est', 'gt1l/geolocation/tx_pulse_width_lower', 'gt1l/geolocation/tx_pulse_width_upper', 'gt1l/geolocation/velocity_sc', 'gt1l/geolocation/yaw', 'gt1l/geophys_corr/dac', 'gt1l/geophys_corr/delta_time', 'gt1l/geophys_corr/dem_flag', 'gt1l/geophys_corr/dem_h', 'gt1l/geophys_corr/geoid', 'gt1l/geophys_corr/geoid_free2mean', 'gt1l/geophys_corr/tide_earth', 'gt1l/geophys_corr/tide_earth_free2mean', 'gt1l/geophys_corr/tide_equilibrium', 'gt1l/geophys_corr/tide_load', 'gt1l/geophys_corr/tide_oc_pole', 'gt1l/geophys_corr/tide_ocean', 'gt1l/geophys_corr/tide_pole', 'gt1l/heights/delta_time', 'gt1l/heights/dist_ph_across', 'gt1l/heights/dist_ph_along', 'gt1l/heights/h_ph', 'gt1l/heights/lat_ph', 'gt1l/heights/lon_ph', 'gt1l/heights/pce_mframe_cnt', 'gt1l/heights/ph_id_channel', 'gt1l/heights/ph_id_count', 'gt1l/heights/ph_id_pulse', 'gt1l/heights/quality_ph', 'gt1l/heights/signal_conf_ph', 'gt1l/signal_find_output/inlandwater/bckgrd_mean', 'gt1l/signal_find_output/inlandwater/bckgrd_sigma', 'gt1l/signal_find_output/inlandwater/delta_time', 'gt1l/signal_find_output/inlandwater/t_pc_delta', 'gt1l/signal_find_output/inlandwater/z_pc_delta', 'gt1l/signal_find_output/land/bckgrd_mean', 'gt1l/signal_find_output/land/bckgrd_sigma', 'gt1l/signal_find_output/land/delta_time', 'gt1l/signal_find_output/land/t_pc_delta', 'gt1l/signal_find_output/land/z_pc_delta', 'gt1l/signal_find_output/land_ice/bckgrd_mean', 'gt1l/signal_find_output/land_ice/bckgrd_sigma', 'gt1l/signal_find_output/land_ice/delta_time', 'gt1l/signal_find_output/land_ice/t_pc_delta', 'gt1l/signal_find_output/land_ice/z_pc_delta', 'gt1l/signal_find_output/ocean/bckgrd_mean', 'gt1l/signal_find_output/ocean/bckgrd_sigma', 'gt1l/signal_find_output/ocean/delta_time', 'gt1l/signal_find_output/ocean/t_pc_delta', 'gt1l/signal_find_output/ocean/z_pc_delta', 'gt1l/signal_find_output/sea_ice/bckgrd_mean', 'gt1l/signal_find_output/sea_ice/bckgrd_sigma', 'gt1l/signal_find_output/sea_ice/delta_time', 'gt1l/signal_find_output/sea_ice/t_pc_delta', 'gt1l/signal_find_output/sea_ice/z_pc_delta', 'gt1r/bckgrd_atlas/bckgrd_counts', 'gt1r/bckgrd_atlas/bckgrd_counts_reduced', 'gt1r/bckgrd_atlas/bckgrd_hist_top', 'gt1r/bckgrd_atlas/bckgrd_int_height', 'gt1r/bckgrd_atlas/bckgrd_int_height_reduced', 'gt1r/bckgrd_atlas/bckgrd_rate', 'gt1r/bckgrd_atlas/delta_time', 'gt1r/bckgrd_atlas/pce_mframe_cnt', 'gt1r/bckgrd_atlas/tlm_height_band1', 'gt1r/bckgrd_atlas/tlm_height_band2', 'gt1r/bckgrd_atlas/tlm_top_band1', 'gt1r/bckgrd_atlas/tlm_top_band2', 'gt1r/geolocation/altitude_sc', 'gt1r/geolocation/bounce_time_offset', 'gt1r/geolocation/delta_time', 'gt1r/geolocation/full_sat_fract', 'gt1r/geolocation/near_sat_fract', 'gt1r/geolocation/neutat_delay_derivative', 'gt1r/geolocation/neutat_delay_total', 'gt1r/geolocation/neutat_ht', 'gt1r/geolocation/ph_index_beg', 'gt1r/geolocation/pitch', 'gt1r/geolocation/podppd_flag', 'gt1r/geolocation/range_bias_corr', 'gt1r/geolocation/ref_azimuth', 'gt1r/geolocation/ref_elev', 'gt1r/geolocation/reference_photon_index', 'gt1r/geolocation/reference_photon_lat', 'gt1r/geolocation/reference_photon_lon', 'gt1r/geolocation/roll', 'gt1r/geolocation/segment_dist_x', 'gt1r/geolocation/segment_id', 'gt1r/geolocation/segment_length', 'gt1r/geolocation/segment_ph_cnt', 'gt1r/geolocation/sigma_across', 'gt1r/geolocation/sigma_along', 'gt1r/geolocation/sigma_h', 'gt1r/geolocation/sigma_lat', 'gt1r/geolocation/sigma_lon', 'gt1r/geolocation/solar_azimuth', 'gt1r/geolocation/solar_elevation', 'gt1r/geolocation/surf_type', 'gt1r/geolocation/tx_pulse_energy', 'gt1r/geolocation/tx_pulse_skew_est', 'gt1r/geolocation/tx_pulse_width_lower', 'gt1r/geolocation/tx_pulse_width_upper', 'gt1r/geolocation/velocity_sc', 'gt1r/geolocation/yaw', 'gt1r/geophys_corr/dac', 'gt1r/geophys_corr/delta_time', 'gt1r/geophys_corr/dem_flag', 'gt1r/geophys_corr/dem_h', 'gt1r/geophys_corr/geoid', 'gt1r/geophys_corr/geoid_free2mean', 'gt1r/geophys_corr/tide_earth', 'gt1r/geophys_corr/tide_earth_free2mean', 'gt1r/geophys_corr/tide_equilibrium', 'gt1r/geophys_corr/tide_load', 'gt1r/geophys_corr/tide_oc_pole', 'gt1r/geophys_corr/tide_ocean', 'gt1r/geophys_corr/tide_pole', 'gt1r/heights/delta_time', 'gt1r/heights/dist_ph_across', 'gt1r/heights/dist_ph_along', 'gt1r/heights/h_ph', 'gt1r/heights/lat_ph', 'gt1r/heights/lon_ph', 'gt1r/heights/pce_mframe_cnt', 'gt1r/heights/ph_id_channel', 'gt1r/heights/ph_id_count', 'gt1r/heights/ph_id_pulse', 'gt1r/heights/quality_ph', 'gt1r/heights/signal_conf_ph', 'gt1r/signal_find_output/inlandwater/bckgrd_mean', 'gt1r/signal_find_output/inlandwater/bckgrd_sigma', 'gt1r/signal_find_output/inlandwater/delta_time', 'gt1r/signal_find_output/inlandwater/t_pc_delta', 'gt1r/signal_find_output/inlandwater/z_pc_delta', 'gt1r/signal_find_output/land/bckgrd_mean', 'gt1r/signal_find_output/land/bckgrd_sigma', 'gt1r/signal_find_output/land/delta_time', 'gt1r/signal_find_output/land/t_pc_delta', 'gt1r/signal_find_output/land/z_pc_delta', 'gt1r/signal_find_output/land_ice/bckgrd_mean', 'gt1r/signal_find_output/land_ice/bckgrd_sigma', 'gt1r/signal_find_output/land_ice/delta_time', 'gt1r/signal_find_output/land_ice/t_pc_delta', 'gt1r/signal_find_output/land_ice/z_pc_delta', 'gt1r/signal_find_output/ocean/bckgrd_mean', 'gt1r/signal_find_output/ocean/bckgrd_sigma', 'gt1r/signal_find_output/ocean/delta_time', 'gt1r/signal_find_output/ocean/t_pc_delta', 'gt1r/signal_find_output/ocean/z_pc_delta', 'gt1r/signal_find_output/sea_ice/bckgrd_mean', 'gt1r/signal_find_output/sea_ice/bckgrd_sigma', 'gt1r/signal_find_output/sea_ice/delta_time', 'gt1r/signal_find_output/sea_ice/t_pc_delta', 'gt1r/signal_find_output/sea_ice/z_pc_delta', 'gt2l/bckgrd_atlas/bckgrd_counts', 'gt2l/bckgrd_atlas/bckgrd_counts_reduced', 'gt2l/bckgrd_atlas/bckgrd_hist_top', 'gt2l/bckgrd_atlas/bckgrd_int_height', 'gt2l/bckgrd_atlas/bckgrd_int_height_reduced', 'gt2l/bckgrd_atlas/bckgrd_rate', 'gt2l/bckgrd_atlas/delta_time', 'gt2l/bckgrd_atlas/pce_mframe_cnt', 'gt2l/bckgrd_atlas/tlm_height_band1', 'gt2l/bckgrd_atlas/tlm_height_band2', 'gt2l/bckgrd_atlas/tlm_top_band1', 'gt2l/bckgrd_atlas/tlm_top_band2', 'gt2l/geolocation/altitude_sc', 'gt2l/geolocation/bounce_time_offset', 'gt2l/geolocation/delta_time', 'gt2l/geolocation/full_sat_fract', 'gt2l/geolocation/near_sat_fract', 'gt2l/geolocation/neutat_delay_derivative', 'gt2l/geolocation/neutat_delay_total', 'gt2l/geolocation/neutat_ht', 'gt2l/geolocation/ph_index_beg', 'gt2l/geolocation/pitch', 'gt2l/geolocation/podppd_flag', 'gt2l/geolocation/range_bias_corr', 'gt2l/geolocation/ref_azimuth', 'gt2l/geolocation/ref_elev', 'gt2l/geolocation/reference_photon_index', 'gt2l/geolocation/reference_photon_lat', 'gt2l/geolocation/reference_photon_lon', 'gt2l/geolocation/roll', 'gt2l/geolocation/segment_dist_x', 'gt2l/geolocation/segment_id', 'gt2l/geolocation/segment_length', 'gt2l/geolocation/segment_ph_cnt', 'gt2l/geolocation/sigma_across', 'gt2l/geolocation/sigma_along', 'gt2l/geolocation/sigma_h', 'gt2l/geolocation/sigma_lat', 'gt2l/geolocation/sigma_lon', 'gt2l/geolocation/solar_azimuth', 'gt2l/geolocation/solar_elevation', 'gt2l/geolocation/surf_type', 'gt2l/geolocation/tx_pulse_energy', 'gt2l/geolocation/tx_pulse_skew_est', 'gt2l/geolocation/tx_pulse_width_lower', 'gt2l/geolocation/tx_pulse_width_upper', 'gt2l/geolocation/velocity_sc', 'gt2l/geolocation/yaw', 'gt2l/geophys_corr/dac', 'gt2l/geophys_corr/delta_time', 'gt2l/geophys_corr/dem_flag', 'gt2l/geophys_corr/dem_h', 'gt2l/geophys_corr/geoid', 'gt2l/geophys_corr/geoid_free2mean', 'gt2l/geophys_corr/tide_earth', 'gt2l/geophys_corr/tide_earth_free2mean', 'gt2l/geophys_corr/tide_equilibrium', 'gt2l/geophys_corr/tide_load', 'gt2l/geophys_corr/tide_oc_pole', 'gt2l/geophys_corr/tide_ocean', 'gt2l/geophys_corr/tide_pole', 'gt2l/heights/delta_time', 'gt2l/heights/dist_ph_across', 'gt2l/heights/dist_ph_along', 'gt2l/heights/h_ph', 'gt2l/heights/lat_ph', 'gt2l/heights/lon_ph', 'gt2l/heights/pce_mframe_cnt', 'gt2l/heights/ph_id_channel', 'gt2l/heights/ph_id_count', 'gt2l/heights/ph_id_pulse', 'gt2l/heights/quality_ph', 'gt2l/heights/signal_conf_ph', 'gt2l/signal_find_output/inlandwater/bckgrd_mean', 'gt2l/signal_find_output/inlandwater/bckgrd_sigma', 'gt2l/signal_find_output/inlandwater/delta_time', 'gt2l/signal_find_output/inlandwater/t_pc_delta', 'gt2l/signal_find_output/inlandwater/z_pc_delta', 'gt2l/signal_find_output/land/bckgrd_mean', 'gt2l/signal_find_output/land/bckgrd_sigma', 'gt2l/signal_find_output/land/delta_time', 'gt2l/signal_find_output/land/t_pc_delta', 'gt2l/signal_find_output/land/z_pc_delta', 'gt2l/signal_find_output/land_ice/bckgrd_mean', 'gt2l/signal_find_output/land_ice/bckgrd_sigma', 'gt2l/signal_find_output/land_ice/delta_time', 'gt2l/signal_find_output/land_ice/t_pc_delta', 'gt2l/signal_find_output/land_ice/z_pc_delta', 'gt2l/signal_find_output/ocean/bckgrd_mean', 'gt2l/signal_find_output/ocean/bckgrd_sigma', 'gt2l/signal_find_output/ocean/delta_time', 'gt2l/signal_find_output/ocean/t_pc_delta', 'gt2l/signal_find_output/ocean/z_pc_delta', 'gt2l/signal_find_output/sea_ice/bckgrd_mean', 'gt2l/signal_find_output/sea_ice/bckgrd_sigma', 'gt2l/signal_find_output/sea_ice/delta_time', 'gt2l/signal_find_output/sea_ice/t_pc_delta', 'gt2l/signal_find_output/sea_ice/z_pc_delta', 'gt2r/bckgrd_atlas/bckgrd_counts', 'gt2r/bckgrd_atlas/bckgrd_counts_reduced', 'gt2r/bckgrd_atlas/bckgrd_hist_top', 'gt2r/bckgrd_atlas/bckgrd_int_height', 'gt2r/bckgrd_atlas/bckgrd_int_height_reduced', 'gt2r/bckgrd_atlas/bckgrd_rate', 'gt2r/bckgrd_atlas/delta_time', 'gt2r/bckgrd_atlas/pce_mframe_cnt', 'gt2r/bckgrd_atlas/tlm_height_band1', 'gt2r/bckgrd_atlas/tlm_height_band2', 'gt2r/bckgrd_atlas/tlm_top_band1', 'gt2r/bckgrd_atlas/tlm_top_band2', 'gt2r/geolocation/altitude_sc', 'gt2r/geolocation/bounce_time_offset', 'gt2r/geolocation/delta_time', 'gt2r/geolocation/full_sat_fract', 'gt2r/geolocation/near_sat_fract', 'gt2r/geolocation/neutat_delay_derivative', 'gt2r/geolocation/neutat_delay_total', 'gt2r/geolocation/neutat_ht', 'gt2r/geolocation/ph_index_beg', 'gt2r/geolocation/pitch', 'gt2r/geolocation/podppd_flag', 'gt2r/geolocation/range_bias_corr', 'gt2r/geolocation/ref_azimuth', 'gt2r/geolocation/ref_elev', 'gt2r/geolocation/reference_photon_index', 'gt2r/geolocation/reference_photon_lat', 'gt2r/geolocation/reference_photon_lon', 'gt2r/geolocation/roll', 'gt2r/geolocation/segment_dist_x', 'gt2r/geolocation/segment_id', 'gt2r/geolocation/segment_length', 'gt2r/geolocation/segment_ph_cnt', 'gt2r/geolocation/sigma_across', 'gt2r/geolocation/sigma_along', 'gt2r/geolocation/sigma_h', 'gt2r/geolocation/sigma_lat', 'gt2r/geolocation/sigma_lon', 'gt2r/geolocation/solar_azimuth', 'gt2r/geolocation/solar_elevation', 'gt2r/geolocation/surf_type', 'gt2r/geolocation/tx_pulse_energy', 'gt2r/geolocation/tx_pulse_skew_est', 'gt2r/geolocation/tx_pulse_width_lower', 'gt2r/geolocation/tx_pulse_width_upper', 'gt2r/geolocation/velocity_sc', 'gt2r/geolocation/yaw', 'gt2r/geophys_corr/dac', 'gt2r/geophys_corr/delta_time', 'gt2r/geophys_corr/dem_flag', 'gt2r/geophys_corr/dem_h', 'gt2r/geophys_corr/geoid', 'gt2r/geophys_corr/geoid_free2mean', 'gt2r/geophys_corr/tide_earth', 'gt2r/geophys_corr/tide_earth_free2mean', 'gt2r/geophys_corr/tide_equilibrium', 'gt2r/geophys_corr/tide_load', 'gt2r/geophys_corr/tide_oc_pole', 'gt2r/geophys_corr/tide_ocean', 'gt2r/geophys_corr/tide_pole', 'gt2r/heights/delta_time', 'gt2r/heights/dist_ph_across', 'gt2r/heights/dist_ph_along', 'gt2r/heights/h_ph', 'gt2r/heights/lat_ph', 'gt2r/heights/lon_ph', 'gt2r/heights/pce_mframe_cnt', 'gt2r/heights/ph_id_channel', 'gt2r/heights/ph_id_count', 'gt2r/heights/ph_id_pulse', 'gt2r/heights/quality_ph', 'gt2r/heights/signal_conf_ph', 'gt2r/signal_find_output/inlandwater/bckgrd_mean', 'gt2r/signal_find_output/inlandwater/bckgrd_sigma', 'gt2r/signal_find_output/inlandwater/delta_time', 'gt2r/signal_find_output/inlandwater/t_pc_delta', 'gt2r/signal_find_output/inlandwater/z_pc_delta', 'gt2r/signal_find_output/land/bckgrd_mean', 'gt2r/signal_find_output/land/bckgrd_sigma', 'gt2r/signal_find_output/land/delta_time', 'gt2r/signal_find_output/land/t_pc_delta', 'gt2r/signal_find_output/land/z_pc_delta', 'gt2r/signal_find_output/land_ice/bckgrd_mean', 'gt2r/signal_find_output/land_ice/bckgrd_sigma', 'gt2r/signal_find_output/land_ice/delta_time', 'gt2r/signal_find_output/land_ice/t_pc_delta', 'gt2r/signal_find_output/land_ice/z_pc_delta', 'gt2r/signal_find_output/ocean/bckgrd_mean', 'gt2r/signal_find_output/ocean/bckgrd_sigma', 'gt2r/signal_find_output/ocean/delta_time', 'gt2r/signal_find_output/ocean/t_pc_delta', 'gt2r/signal_find_output/ocean/z_pc_delta', 'gt2r/signal_find_output/sea_ice/bckgrd_mean', 'gt2r/signal_find_output/sea_ice/bckgrd_sigma', 'gt2r/signal_find_output/sea_ice/delta_time', 'gt2r/signal_find_output/sea_ice/t_pc_delta', 'gt2r/signal_find_output/sea_ice/z_pc_delta', 'gt3l/bckgrd_atlas/bckgrd_counts', 'gt3l/bckgrd_atlas/bckgrd_counts_reduced', 'gt3l/bckgrd_atlas/bckgrd_hist_top', 'gt3l/bckgrd_atlas/bckgrd_int_height', 'gt3l/bckgrd_atlas/bckgrd_int_height_reduced', 'gt3l/bckgrd_atlas/bckgrd_rate', 'gt3l/bckgrd_atlas/delta_time', 'gt3l/bckgrd_atlas/pce_mframe_cnt', 'gt3l/bckgrd_atlas/tlm_height_band1', 'gt3l/bckgrd_atlas/tlm_height_band2', 'gt3l/bckgrd_atlas/tlm_top_band1', 'gt3l/bckgrd_atlas/tlm_top_band2', 'gt3l/geolocation/altitude_sc', 'gt3l/geolocation/bounce_time_offset', 'gt3l/geolocation/delta_time', 'gt3l/geolocation/full_sat_fract', 'gt3l/geolocation/near_sat_fract', 'gt3l/geolocation/neutat_delay_derivative', 'gt3l/geolocation/neutat_delay_total', 'gt3l/geolocation/neutat_ht', 'gt3l/geolocation/ph_index_beg', 'gt3l/geolocation/pitch', 'gt3l/geolocation/podppd_flag', 'gt3l/geolocation/range_bias_corr', 'gt3l/geolocation/ref_azimuth', 'gt3l/geolocation/ref_elev', 'gt3l/geolocation/reference_photon_index', 'gt3l/geolocation/reference_photon_lat', 'gt3l/geolocation/reference_photon_lon', 'gt3l/geolocation/roll', 'gt3l/geolocation/segment_dist_x', 'gt3l/geolocation/segment_id', 'gt3l/geolocation/segment_length', 'gt3l/geolocation/segment_ph_cnt', 'gt3l/geolocation/sigma_across', 'gt3l/geolocation/sigma_along', 'gt3l/geolocation/sigma_h', 'gt3l/geolocation/sigma_lat', 'gt3l/geolocation/sigma_lon', 'gt3l/geolocation/solar_azimuth', 'gt3l/geolocation/solar_elevation', 'gt3l/geolocation/surf_type', 'gt3l/geolocation/tx_pulse_energy', 'gt3l/geolocation/tx_pulse_skew_est', 'gt3l/geolocation/tx_pulse_width_lower', 'gt3l/geolocation/tx_pulse_width_upper', 'gt3l/geolocation/velocity_sc', 'gt3l/geolocation/yaw', 'gt3l/geophys_corr/dac', 'gt3l/geophys_corr/delta_time', 'gt3l/geophys_corr/dem_flag', 'gt3l/geophys_corr/dem_h', 'gt3l/geophys_corr/geoid', 'gt3l/geophys_corr/geoid_free2mean', 'gt3l/geophys_corr/tide_earth', 'gt3l/geophys_corr/tide_earth_free2mean', 'gt3l/geophys_corr/tide_equilibrium', 'gt3l/geophys_corr/tide_load', 'gt3l/geophys_corr/tide_oc_pole', 'gt3l/geophys_corr/tide_ocean', 'gt3l/geophys_corr/tide_pole', 'gt3l/heights/delta_time', 'gt3l/heights/dist_ph_across', 'gt3l/heights/dist_ph_along', 'gt3l/heights/h_ph', 'gt3l/heights/lat_ph', 'gt3l/heights/lon_ph', 'gt3l/heights/pce_mframe_cnt', 'gt3l/heights/ph_id_channel', 'gt3l/heights/ph_id_count', 'gt3l/heights/ph_id_pulse', 'gt3l/heights/quality_ph', 'gt3l/heights/signal_conf_ph', 'gt3l/signal_find_output/inlandwater/bckgrd_mean', 'gt3l/signal_find_output/inlandwater/bckgrd_sigma', 'gt3l/signal_find_output/inlandwater/delta_time', 'gt3l/signal_find_output/inlandwater/t_pc_delta', 'gt3l/signal_find_output/inlandwater/z_pc_delta', 'gt3l/signal_find_output/land/bckgrd_mean', 'gt3l/signal_find_output/land/bckgrd_sigma', 'gt3l/signal_find_output/land/delta_time', 'gt3l/signal_find_output/land/t_pc_delta', 'gt3l/signal_find_output/land/z_pc_delta', 'gt3l/signal_find_output/land_ice/bckgrd_mean', 'gt3l/signal_find_output/land_ice/bckgrd_sigma', 'gt3l/signal_find_output/land_ice/delta_time', 'gt3l/signal_find_output/land_ice/t_pc_delta', 'gt3l/signal_find_output/land_ice/z_pc_delta', 'gt3l/signal_find_output/ocean/bckgrd_mean', 'gt3l/signal_find_output/ocean/bckgrd_sigma', 'gt3l/signal_find_output/ocean/delta_time', 'gt3l/signal_find_output/ocean/t_pc_delta', 'gt3l/signal_find_output/ocean/z_pc_delta', 'gt3l/signal_find_output/sea_ice/bckgrd_mean', 'gt3l/signal_find_output/sea_ice/bckgrd_sigma', 'gt3l/signal_find_output/sea_ice/delta_time', 'gt3l/signal_find_output/sea_ice/t_pc_delta', 'gt3l/signal_find_output/sea_ice/z_pc_delta', 'gt3r/bckgrd_atlas/bckgrd_counts', 'gt3r/bckgrd_atlas/bckgrd_counts_reduced', 'gt3r/bckgrd_atlas/bckgrd_hist_top', 'gt3r/bckgrd_atlas/bckgrd_int_height', 'gt3r/bckgrd_atlas/bckgrd_int_height_reduced', 'gt3r/bckgrd_atlas/bckgrd_rate', 'gt3r/bckgrd_atlas/delta_time', 'gt3r/bckgrd_atlas/pce_mframe_cnt', 'gt3r/bckgrd_atlas/tlm_height_band1', 'gt3r/bckgrd_atlas/tlm_height_band2', 'gt3r/bckgrd_atlas/tlm_top_band1', 'gt3r/bckgrd_atlas/tlm_top_band2', 'gt3r/geolocation/altitude_sc', 'gt3r/geolocation/bounce_time_offset', 'gt3r/geolocation/delta_time', 'gt3r/geolocation/full_sat_fract', 'gt3r/geolocation/near_sat_fract', 'gt3r/geolocation/neutat_delay_derivative', 'gt3r/geolocation/neutat_delay_total', 'gt3r/geolocation/neutat_ht', 'gt3r/geolocation/ph_index_beg', 'gt3r/geolocation/pitch', 'gt3r/geolocation/podppd_flag', 'gt3r/geolocation/range_bias_corr', 'gt3r/geolocation/ref_azimuth', 'gt3r/geolocation/ref_elev', 'gt3r/geolocation/reference_photon_index', 'gt3r/geolocation/reference_photon_lat', 'gt3r/geolocation/reference_photon_lon', 'gt3r/geolocation/roll', 'gt3r/geolocation/segment_dist_x', 'gt3r/geolocation/segment_id', 'gt3r/geolocation/segment_length', 'gt3r/geolocation/segment_ph_cnt', 'gt3r/geolocation/sigma_across', 'gt3r/geolocation/sigma_along', 'gt3r/geolocation/sigma_h', 'gt3r/geolocation/sigma_lat', 'gt3r/geolocation/sigma_lon', 'gt3r/geolocation/solar_azimuth', 'gt3r/geolocation/solar_elevation', 'gt3r/geolocation/surf_type', 'gt3r/geolocation/tx_pulse_energy', 'gt3r/geolocation/tx_pulse_skew_est', 'gt3r/geolocation/tx_pulse_width_lower', 'gt3r/geolocation/tx_pulse_width_upper', 'gt3r/geolocation/velocity_sc', 'gt3r/geolocation/yaw', 'gt3r/geophys_corr/dac', 'gt3r/geophys_corr/delta_time', 'gt3r/geophys_corr/dem_flag', 'gt3r/geophys_corr/dem_h', 'gt3r/geophys_corr/geoid', 'gt3r/geophys_corr/geoid_free2mean', 'gt3r/geophys_corr/tide_earth', 'gt3r/geophys_corr/tide_earth_free2mean', 'gt3r/geophys_corr/tide_equilibrium', 'gt3r/geophys_corr/tide_load', 'gt3r/geophys_corr/tide_oc_pole', 'gt3r/geophys_corr/tide_ocean', 'gt3r/geophys_corr/tide_pole', 'gt3r/heights/delta_time', 'gt3r/heights/dist_ph_across', 'gt3r/heights/dist_ph_along', 'gt3r/heights/h_ph', 'gt3r/heights/lat_ph', 'gt3r/heights/lon_ph', 'gt3r/heights/pce_mframe_cnt', 'gt3r/heights/ph_id_channel', 'gt3r/heights/ph_id_count', 'gt3r/heights/ph_id_pulse', 'gt3r/heights/quality_ph', 'gt3r/heights/signal_conf_ph', 'gt3r/signal_find_output/inlandwater/bckgrd_mean', 'gt3r/signal_find_output/inlandwater/bckgrd_sigma', 'gt3r/signal_find_output/inlandwater/delta_time', 'gt3r/signal_find_output/inlandwater/t_pc_delta', 'gt3r/signal_find_output/inlandwater/z_pc_delta', 'gt3r/signal_find_output/land/bckgrd_mean', 'gt3r/signal_find_output/land/bckgrd_sigma', 'gt3r/signal_find_output/land/delta_time', 'gt3r/signal_find_output/land/t_pc_delta', 'gt3r/signal_find_output/land/z_pc_delta', 'gt3r/signal_find_output/land_ice/bckgrd_mean', 'gt3r/signal_find_output/land_ice/bckgrd_sigma', 'gt3r/signal_find_output/land_ice/delta_time', 'gt3r/signal_find_output/land_ice/t_pc_delta', 'gt3r/signal_find_output/land_ice/z_pc_delta', 'gt3r/signal_find_output/ocean/bckgrd_mean', 'gt3r/signal_find_output/ocean/bckgrd_sigma', 'gt3r/signal_find_output/ocean/delta_time', 'gt3r/signal_find_output/ocean/t_pc_delta', 'gt3r/signal_find_output/ocean/z_pc_delta', 'gt3r/signal_find_output/sea_ice/bckgrd_mean', 'gt3r/signal_find_output/sea_ice/bckgrd_sigma', 'gt3r/signal_find_output/sea_ice/delta_time', 'gt3r/signal_find_output/sea_ice/t_pc_delta', 'gt3r/signal_find_output/sea_ice/z_pc_delta', 'orbit_info/crossing_time', 'orbit_info/cycle_number', 'orbit_info/lan', 'orbit_info/orbit_number', 'orbit_info/rgt', 'orbit_info/sc_orient', 'orbit_info/sc_orient_time', 'quality_assessment/delta_time', 'quality_assessment/gt1l/qa_perc_signal_conf_ph_high', 'quality_assessment/gt1l/qa_perc_signal_conf_ph_low', 'quality_assessment/gt1l/qa_perc_signal_conf_ph_med', 'quality_assessment/gt1l/qa_perc_surf_type', 'quality_assessment/gt1l/qa_total_signal_conf_ph_high', 'quality_assessment/gt1l/qa_total_signal_conf_ph_low', 'quality_assessment/gt1l/qa_total_signal_conf_ph_med', 'quality_assessment/gt1r/qa_perc_signal_conf_ph_high', 'quality_assessment/gt1r/qa_perc_signal_conf_ph_low', 'quality_assessment/gt1r/qa_perc_signal_conf_ph_med', 'quality_assessment/gt1r/qa_perc_surf_type', 'quality_assessment/gt1r/qa_total_signal_conf_ph_high', 'quality_assessment/gt1r/qa_total_signal_conf_ph_low', 'quality_assessment/gt1r/qa_total_signal_conf_ph_med', 'quality_assessment/gt2l/qa_perc_signal_conf_ph_high', 'quality_assessment/gt2l/qa_perc_signal_conf_ph_low', 'quality_assessment/gt2l/qa_perc_signal_conf_ph_med', 'quality_assessment/gt2l/qa_perc_surf_type', 'quality_assessment/gt2l/qa_total_signal_conf_ph_high', 'quality_assessment/gt2l/qa_total_signal_conf_ph_low', 'quality_assessment/gt2l/qa_total_signal_conf_ph_med', 'quality_assessment/gt2r/qa_perc_signal_conf_ph_high', 'quality_assessment/gt2r/qa_perc_signal_conf_ph_low', 'quality_assessment/gt2r/qa_perc_signal_conf_ph_med', 'quality_assessment/gt2r/qa_perc_surf_type', 'quality_assessment/gt2r/qa_total_signal_conf_ph_high', 'quality_assessment/gt2r/qa_total_signal_conf_ph_low', 'quality_assessment/gt2r/qa_total_signal_conf_ph_med', 'quality_assessment/gt3l/qa_perc_signal_conf_ph_high', 'quality_assessment/gt3l/qa_perc_signal_conf_ph_low', 'quality_assessment/gt3l/qa_perc_signal_conf_ph_med', 'quality_assessment/gt3l/qa_perc_surf_type', 'quality_assessment/gt3l/qa_total_signal_conf_ph_high', 'quality_assessment/gt3l/qa_total_signal_conf_ph_low', 'quality_assessment/gt3l/qa_total_signal_conf_ph_med', 'quality_assessment/gt3r/qa_perc_signal_conf_ph_high', 'quality_assessment/gt3r/qa_perc_signal_conf_ph_low', 'quality_assessment/gt3r/qa_perc_signal_conf_ph_med', 'quality_assessment/gt3r/qa_perc_surf_type', 'quality_assessment/gt3r/qa_total_signal_conf_ph_high', 'quality_assessment/gt3r/qa_total_signal_conf_ph_low', 'quality_assessment/gt3r/qa_total_signal_conf_ph_med', 'quality_assessment/qa_granule_fail_reason', 'quality_assessment/qa_granule_pass_fail']
    # B


# # -- iterate over ATLAS major frames
# photon_mframes = val['heights']['pce_mframe_cnt']
# # -- background ATLAS group variables are based upon 50-shot summations
# # -- PCE Major Frames are based upon 200-shot summations
# pce_mframe_cnt = val['bckgrd_atlas']['pce_mframe_cnt']
# # -- find unique major frames and their indices within background ATLAS group
# # -- (there will 4 background ATLAS time steps for nearly every major frame)
# unique_major_frames, unique_index = np.unique(pce_mframe_cnt, return_index=True)
# # -- number of unique major frames in granule for beam
# major_frame_count = len(unique_major_frames)
# # -- height of each telemetry band for a major frame
# tlm_height = {}
# tlm_height['band1'] = val['bckgrd_atlas']['tlm_height_band1']
# tlm_height['band2'] = val['bckgrd_atlas']['tlm_height_band2']
# # -- elevation above ellipsoid of each telemetry band for a major frame
# tlm_top = {}
# tlm_top['band1'] = val['bckgrd_atlas']['tlm_top_band1']
# tlm_top['band2'] = val['bckgrd_atlas']['tlm_top_band2']
# # -- buffer to telemetry band to set as valid
# tlm_buffer = 100.0
# # -- flag denoting photon events as possible TEP
# if (int(RL) < 4):
#     isTEP = np.any((val['heights']['signal_conf_ph'][:] == -2), axis=1)
# else:
#     isTEP = (val['heights']['quality_ph'][:] == 3)
# # -- photon event weights and signal-to-noise ratio
# pe_weights = np.zeros((n_pe), dtype=np.float64)
# Segment_Photon_SNR[gtx] = np.zeros((n_pe), dtype=int)
#
# # -- run for each major frame
# for iteration, idx in enumerate(unique_index):
#     utl.log()
#     # -- photon indices for major frame (buffered by 1 frame on each side)
#     # -- do not use possible TEP photons in photon classification
#     i1, = np.nonzero((photon_mframes >= unique_major_frames[iteration] - 1) &
#                      (photon_mframes <= unique_major_frames[iteration] + 1) &
#                      np.logical_not(isTEP))
#     # -- indices for the major frame within the buffered window
#     i2, = np.nonzero(photon_mframes[i1] == unique_major_frames[iteration])
#
#     # -- sum of telemetry band widths for major frame
#     h_win_width = 0.0
#     # -- check that each telemetry band is close to DEM
#     for b in ['band1', 'band2']:
#         # -- bottom of the telemetry band for major frame
#         tlm_bot_band = tlm_top[b][idx] - tlm_height[b][idx]
#         if np.any((dem_h[i1[i2]] >= (tlm_bot_band - tlm_buffer)) &
#                   (dem_h[i1[i2]] <= (tlm_top[b][idx] + tlm_buffer))):
#             # -- add telemetry height to window width
#             h_win_width += tlm_height[b][idx]
#
#     # -- calculate photon event weights
#     pe_weights[i1[i2]] = classify_photons(x_atc[i1], h_ph[i1],
#                                           h_win_width, i2, K=3, min_ph=3, min_xspread=1.0,
#                                           min_hspread=0.01, aspect=3, method='linear')
#
# # -- photon event weights scaled to a single byte
# weight_ph = np.array(255 * pe_weights, dtype=np.uint8)
# # -- verify photon event weights
# np.clip(weight_ph, 0, 255, out=weight_ph)
#
# # -- allocate for segment means
# fill_value = attrs['geolocation']['sigma_h']['_FillValue']
# # -- mean longitude of each segment high-confidence photons
# Segment_Lon[gtx] = np.ma.zeros((n_seg), fill_value=fill_value)
# Segment_Lon[gtx].data[:] = Segment_Lon[gtx].fill_value
# Segment_Lon[gtx].mask = np.ones((n_seg), dtype=bool)
# # -- mean longitude of each segment high-confidence photons
# Segment_Lat[gtx] = np.ma.zeros((n_seg), fill_value=fill_value)
# Segment_Lat[gtx].data[:] = Segment_Lat[gtx].fill_value
# Segment_Lat[gtx].mask = np.ones((n_seg), dtype=bool)
# # -- mean height of each segment high-confidence photons
# Segment_Elev[gtx] = np.ma.zeros((n_seg), fill_value=fill_value)
# Segment_Elev[gtx].data[:] = Segment_Elev[gtx].fill_value
# Segment_Elev[gtx].mask = np.ones((n_seg), dtype=bool)
# # -- mean time of each segment high-confidence photons
# Segment_Time[gtx] = np.ma.zeros((n_seg), fill_value=fill_value)
# Segment_Time[gtx].data[:] = Segment_Time[gtx].fill_value
# Segment_Time[gtx].mask = np.ones((n_seg), dtype=bool)
#
# # -- iterate over ATL03 segments to calculate 40m means
# # -- in ATL03 1-based indexing: invalid == 0
# # -- here in 0-based indexing: invalid == -1
# segment_indices, = np.nonzero((Segment_Index_begin[gtx][:-1] >= 0) &
#                               (Segment_Index_begin[gtx][1:] >= 0))
# for j in segment_indices:
#     # -- index for segment j
#     idx = Segment_Index_begin[gtx][j]
#     # -- number of photons in segment (use 2 ATL03 segments)
#     c1 = np.copy(Segment_PE_count[gtx][j])
#     c2 = np.copy(Segment_PE_count[gtx][j + 1])
#     cnt = c1 + c2
#     # -- time of each Photon event (PE)
#     segment_delta_times = val['heights']['delta_time'][idx:idx + cnt]
#     gps_seconds = atlas_sdp_gps_epoch + segment_delta_times
#     time_leaps = icesat2_toolkit.time.count_leap_seconds(gps_seconds)
#     # -- Photon event lat/lon and elevation (WGS84)
#     segment_heights = h_ph[idx:idx + cnt].copy()
#     segment_lats = val['heights']['lat_ph'][idx:idx + cnt]
#     segment_lons = val['heights']['lon_ph'][idx:idx + cnt]
#     # -- calculate segment time in Julian days (UTC)
#     segment_times = 2444244.5 + (gps_seconds - time_leaps) / 86400.0
#     # -- Photon event channel and identification
#     ID_channel = val['heights']['ph_id_channel'][idx:idx + cnt]
#     ID_pulse = val['heights']['ph_id_pulse'][idx:idx + cnt]
#     n_pulses = np.unique(ID_pulse).__len__()
#     frame_number = val['heights']['pce_mframe_cnt'][idx:idx + cnt]
#     # -- along-track X and Y coordinates
#     distance_along_X = np.copy(x_atc[idx:idx + cnt])
#     distance_along_Y = np.copy(y_atc[idx:idx + cnt])
#     # -- check the spread of photons along-track (must be > 20m)
#     along_X_spread = distance_along_X.max() - distance_along_X.min()
#     # -- Along-track distance between 2 segments
#     X_atc = Segment_Distance[gtx][j] + Segment_Length[gtx][j]
#     # -- check confidence level associated with each photon event
#     # -- -2: TEP
#     # -- -1: Events not associated with a specific surface type
#     # --  0: noise
#     # --  1: buffer but algorithm classifies as background
#     # --  2: low
#     # --  3: medium
#     # --  4: high
#     # -- Signal classification confidence for land ice
#     # -- 0=Land; 1=Ocean; 2=SeaIce; 3=LandIce; 4=InlandWater
#     ice_sig_conf = val['heights']['signal_conf_ph'][idx:idx + cnt, 3]
#     ice_sig_low_count = np.count_nonzero(ice_sig_conf >= 1)
#     # -- indices of TEP classified photons
#     ice_sig_tep_pe, = np.nonzero(ice_sig_conf == -2)
#     # -- photon event weights from photon classifier
#     segment_weights = weight_ph[idx:idx + cnt]
#     snr_norm = np.max(segment_weights)
#     # -- photon event signal-to-noise ratio from photon classifier
#     photon_snr = np.zeros((cnt), dtype=int)
#     if (snr_norm > 0):
#         photon_snr[:] = 100.0 * segment_weights / snr_norm
#     # -- copy signal to noise ratio for photons
#     Segment_Photon_SNR[gtx][idx:idx + cnt] = np.copy(photon_snr)
#     # -- photon confidence levels from classifier
#     pe_sig_conf = np.zeros((cnt), dtype=int)
#     # -- calculate confidence levels from photon classifier
#     pe_sig_conf[photon_snr >= 25] = 2
#     pe_sig_conf[photon_snr >= 60] = 3
#     pe_sig_conf[photon_snr >= 80] = 4
#     # -- copy classification for TEP photons
#     pe_sig_conf[ice_sig_tep_pe] = -2
#     pe_sig_low_count = np.count_nonzero(pe_sig_conf > 1)
#     # -- check if segment has photon events classified
#     # -- for land ice that are at least low-confidence threshold
#     # -- and that the spread of photons is greater than 20m
#     if (pe_sig_low_count > 10) & (along_X_spread > 20):
#         # -- find photon events that are high-confidence
#         ii, = np.nonzero(pe_sig_conf >= 4)
#         # -- calculate mean elevation (without iterations)
#         # -- NOTE that the segment elevations will NOT be corrected
#         # -- for transmit pulse shape biases or first photon biases
#         Segment_Elev[gtx].data[j] = icesat2_toolkit.fit.fit_geolocation(
#             segment_heights[ii], distance_along_X[ii], X_atc)
#         Segment_Elev[gtx].mask[j] = False
#         # -- calculate geolocation and time of 40m segment center
#         Segment_Lon[gtx].data[j] = icesat2_toolkit.fit.fit_geolocation(
#             segment_lons[ii], distance_along_X[ii], X_atc)
#         Segment_Lon[gtx].mask[j] = False
#         Segment_Lat[gtx].data[j] = icesat2_toolkit.fit.fit_geolocation(
#             segment_lats[ii], distance_along_X[ii], X_atc)
#         Segment_Lat[gtx].mask[j] = False
#         Segment_Time[gtx].data[j] = icesat2_toolkit.fit.fit_geolocation(
#             segment_times[ii], distance_along_X[ii], X_atc)
#         Segment_Time[gtx].mask[j] = False
# # -- clear photon classifier variables for beam group
# pe_weights = None

