# sl:name bg-remove
# sl:description Compare high-quality background removal and matting methods
# sl:app background-removal
# sl:input 1
# sl:output 2
# sl:setup-version 2
# sl:memcheck

_podcrumbs_prepare_repo() {
  local root="$SL_CACHE_DIR/podcrumbs"
  local repo="$root/repo"
  local ref="${PODCRUMBS_REF:-agent/initial-app-catalog}"
  mkdir -p "$root/envs" "$SL_CACHE_DIR/huggingface"
  if [[ ! -d "$repo/.git" ]]; then
    git clone --quiet --depth 1 --no-tags https://github.com/markwelshboy/podcrumbs.git "$repo"
  fi
  git -C "$repo" fetch --quiet --depth 1 --no-tags origin "$ref"
  git -C "$repo" checkout --quiet --detach FETCH_HEAD
}

sl_prepare() {
  _podcrumbs_prepare_repo
}

sl_setup() {
  local repo="$SL_CACHE_DIR/podcrumbs/repo"
  local app="$repo/apps/background-removal"
  local venv="$SL_CACHE_DIR/podcrumbs/envs/background-removal"
  cd "$app"
  VENV="$venv" ./bootstrap.sh
}

sl_run() {
  local repo="$SL_CACHE_DIR/podcrumbs/repo"
  local app="$repo/apps/background-removal"
  local py="$SL_CACHE_DIR/podcrumbs/envs/background-removal/bin/python"
  export HF_HOME="$SL_CACHE_DIR/huggingface"
  cd "$app"
  "$py" run.py "$SL_ARG_1" "$SL_ARG_2" "${SL_EXTRA_ARGS[@]}"
}
