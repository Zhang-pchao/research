# Below are executing commands
nvidia-smi dmon -s pucvmte -o T > nvdmon_job-$SLURM_JOB_ID.log &
module load lammps/29aug2024-dpv3

set -euo pipefail

# ---------- USER SETTINGS ----------
ZMIN=14
ZMAX=100
DZ=1

TEMPLATE_PLUMED="plumed_template.dat"
LAMMPS_IN="in.lammps"

START_RESTART="$(pwd)/start_restart.lmp"

LAMMPS_CMD=(mpirun -np "${SLURM_NTASKS}" -map-by "${MAP_OPT}" lmp -sf omp -pk omp "${OMP_NUM_THREADS}" -in "../${LAMMPS_IN}")
# ----------------------------------

timestamp() { date "+%F %T"; }

# Track current running window so we can mark it FAILED if the job is interrupted
current_wdir=""
current_z0=""

mark_status() {
  local wdir="$1"
  local state="$2"   # RUNNING / DONE / FAILED
  local msg="${3:-}"
  printf "%s %s z0=%s %s\n" "${state}" "$(timestamp)" "${current_z0:-NA}" "${msg}" > "${wdir}/STATUS"
}

on_exit() {
  # If we were running a window and got killed, leave a FAILED mark
  if [[ -n "${current_wdir}" && -f "${current_wdir}/STATUS" ]]; then
    if grep -q '^RUNNING' "${current_wdir}/STATUS"; then
      mark_status "${current_wdir}" "FAILED" "job interrupted (SIG/EXIT)"
    fi
  fi
}
trap on_exit EXIT INT TERM

# Quick sanity checks
[[ -f "${TEMPLATE_PLUMED}" ]] || { echo "ERROR: missing ${TEMPLATE_PLUMED}"; exit 1; }
[[ -f "${LAMMPS_IN}" ]]       || { echo "ERROR: missing ${LAMMPS_IN}"; exit 1; }
[[ -f "${START_RESTART}" ]]   || { echo "ERROR: missing ${START_RESTART}"; exit 1; }

prev_restart="${START_RESTART}"

i=0
for z0 in $(seq ${ZMIN} ${DZ} ${ZMAX}); do
  i=$((i+1))
  wdir=$(printf "win_%03d" "${i}")
  mkdir -p "${wdir}"
  status_file="${wdir}/STATUS"
  restart_out="$(pwd)/${wdir}/restart_out.lmp"

  # ---- Resume logic: skip DONE windows ----
  if [[ -f "${status_file}" ]]; then
    state="$(awk 'NR==1{print $1}' "${status_file}")"
  else
    state=""
  fi

  # If marked DONE but restart_out missing, treat as FAILED and rerun
  if [[ "${state}" == "DONE" && ! -f "${restart_out}" ]]; then
    echo "[WARN] ${wdir} marked DONE but restart_out.lmp missing -> set to FAILED and rerun"
    state="FAILED"
    current_z0="${z0}.0"
    mark_status "${wdir}" "FAILED" "DONE but restart_out missing"
  fi

  if [[ "${state}" == "DONE" ]]; then
    echo "=== Skipping ${wdir} (DONE) z0=${z0}.0 A ==="
    prev_restart="${restart_out}"
    continue
  fi

  # If left as RUNNING (previous crash), convert to FAILED then rerun
  if [[ "${state}" == "RUNNING" ]]; then
    echo "[WARN] ${wdir} was RUNNING from previous job -> mark FAILED and rerun"
    current_z0="${z0}.0"
    mark_status "${wdir}" "FAILED" "stale RUNNING from previous job"
  fi

  # Ensure we have a valid previous restart to start from
  [[ -f "${prev_restart}" ]] || { echo "ERROR: prev restart not found: ${prev_restart}"; exit 1; }

  # ---- Prepare window inputs ----
  sed "s/__Z0__/${z0}.0/g" "${TEMPLATE_PLUMED}" > "${wdir}/plumed.dat"
  ln -sf "${prev_restart}" "${wdir}/restart_in.lmp"

  echo "=== Running ${wdir} z0=${z0}.0 A (prev_restart=$(basename "${prev_restart}")) ==="

  # Mark as RUNNING
  current_wdir="$(pwd)/${wdir}"
  current_z0="${z0}.0"
  mark_status "${wdir}" "RUNNING" "start"

  # Run LAMMPS (capture return code without killing the whole script prematurely)
  set +e
  (
    cd "${wdir}"
    "${LAMMPS_CMD[@]}" > log.lammps
  )
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "ERROR: ${wdir} failed with exit code ${rc}"
    mark_status "${wdir}" "FAILED" "lammps exit code ${rc}"
    exit 1
  fi

  if [[ ! -f "${restart_out}" ]]; then
    echo "ERROR: ${wdir} finished but did not produce restart_out.lmp"
    mark_status "${wdir}" "FAILED" "missing restart_out.lmp"
    exit 1
  fi

  # Mark DONE
  mark_status "${wdir}" "DONE" "ok"
  echo "=== Completed ${wdir} z0=${z0}.0 A ==="

  # Update prev_restart for next window
  prev_restart="${restart_out}"

  # Clear current window tracking
  current_wdir=""
  current_z0=""
done

exit
