#!/usr/bin/env bash

uv_version=${UV_BOOTSTRAP_VERSION:-0.11.28}
uv_install_dir=${UV_BOOTSTRAP_DIR:-"${HOME}/.local/bin"}

if [[ -n ${UV_BIN:-} ]]; then
  uv_bin=$UV_BIN
  if [[ ! -x "$uv_bin" ]]; then
    echo "UV_BIN is not executable: ${uv_bin}" >&2
    return 1
  fi
elif [[ -x "${uv_install_dir}/uv" ]]; then
  uv_bin=${uv_install_dir}/uv
elif uv_bin=$(command -v uv); then
  :
else
  if ! command -v curl >/dev/null; then
    echo "uv is not installed and curl is unavailable for bootstrapping it." >&2
    return 1
  fi

  echo "uv not found; installing uv ${uv_version} in ${uv_install_dir}"
  curl -LsSf "https://astral.sh/uv/${uv_version}/install.sh" |
    env UV_UNMANAGED_INSTALL="$uv_install_dir" sh
  uv_bin=${uv_install_dir}/uv
fi

if [[ ! -x "$uv_bin" ]]; then
  echo "uv bootstrap did not produce an executable at ${uv_bin}." >&2
  return 1
fi

export UV_BIN=$uv_bin
echo "Using $("$UV_BIN" --version) at ${UV_BIN}"
