# sl:name bg-blur
# sl:description Depth-aware background blur with matte-protected subject edges
# sl:input 1
# sl:output 2
# sl:setup-version 1
# sl:memcheck

_podcrumbs_prepare_repo() {
  local root="$SL_CACHE_DIR/podcrumbs"
  local repo="$root/repo"
  local ref="${PODCRUMBS_REF:-main}"
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
  local app="$repo/apps/background-blur"
  local venv="$SL_CACHE_DIR/podcrumbs/envs/background-blur"
  cd "$app"
  VENV="$venv" ./bootstrap.sh
}

sl_run() {
  local repo="$SL_CACHE_DIR/podcrumbs/repo"
  local app="$repo/apps/background-blur"
  local py="$SL_CACHE_DIR/podcrumbs/envs/background-blur/bin/python"
  export HF_HOME="$SL_CACHE_DIR/huggingface"
  cd "$app"
  "$py" background_blur.py "$SL_ARG_1" "$SL_ARG_2" "${SL_EXTRA_ARGS[@]}"
}
