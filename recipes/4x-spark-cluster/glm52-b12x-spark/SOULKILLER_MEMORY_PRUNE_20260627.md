# Soulkiller memory prune - 2026-06-27

Context: during a live GLM-5.2 MTP1/DCP4/128K run, soulkiller was the worst node for active swap interaction. The Ray GPU worker on soulkiller grew from ~1.30GB to ~1.65GB `VmSwap` during one 512-token decode and incurred +1,636 major faults, while the other workers were much lower.

## Baseline comparison

Idle, no model active:

| Host | MemAvailable | SwapCached | SwapFree | Notable overhead |
| --- | ---: | ---: | ---: | --- |
| relic | ~119.1 GiB | 0 MiB | 16.0 GiB | Codex/app tooling; no swapped processes after drain |
| soulkiller | ~120.2 GiB | 87 MiB | 15.78 GiB | bloated `polkitd`, large journald, desktop/audio/snap user session |
| cynosure | ~120.0 GiB | 76 MiB | 15.84 GiB | normal daemon residue, snap desktop helper |
| blackwall | ~120.3 GiB | 71 MiB | 15.86 GiB | normal daemon residue, snap desktop helper |

Soulkiller-specific offenders before pruning:

- `polkitd`: 97MB RSS and ~75MB swapped.
- `systemd-journald`: 92MB RSS, with journals using ~4.0GB on disk.
- User-session desktop/audio helpers: `wireplumber`, `pipewire`, `pipewire-pulse`, `xdg-document-portal`, `xdg-permission-store`, `snapd-desktop-integration`.
- Headless-useless services still active/enabled: `gnome-remote-desktop`, `snapd`, `fwupd`, `apport`, `upower`, `udisks2`, `switcheroo-control`, `rtkit-daemon`.

Network and container runtime were intentionally preserved:

- Kept: `NetworkManager`, `wpa_supplicant`, `docker`, `containerd`, `systemd-resolved`, `sshd`, NVIDIA persistence, RDMA services.

## Changes applied on soulkiller

Masked/stopped system services:

- `gnome-remote-desktop.service`
- `cups.service`, `cups.socket`, `cups.path`, `cups-browsed.service`
- `avahi-daemon.service`, `avahi-daemon.socket`
- `bluetooth.service`
- `ModemManager.service`
- `colord.service`
- `fwupd.service`
- `packagekit.service`
- `apport.service`
- `upower.service`
- `udisks2.service`
- `switcheroo-control.service`
- `rtkit-daemon.service`
- `snapd.service`, `snapd.socket`, `snapd.seeded.service`, `snapd.apparmor.service`

Disabled/masked matt user services when the user manager was present:

- `pipewire.service`, `pipewire.socket`
- `pipewire-pulse.service`, `pipewire-pulse.socket`
- `wireplumber.service`
- `xdg-document-portal.service`
- `xdg-permission-store.service`
- `snap.snapd-desktop-integration.snapd-desktop-integration.service`

Other cleanup:

- Killed leftover desktop/audio helper processes for user `matt`.
- Vacuumed journals to 256MB; this freed ~3.7GB from `/var/log/journal`.
- Restarted `systemd-journald` and `polkit` to drop bloated RSS.
- Set persistent `vm.swappiness=1` in `/etc/sysctl.d/99-kamiwaza-llm-memory.conf`.
- Ran `swapoff -a && swapon -a` and dropped caches with no model active.

## Post-prune state

Post-prune soulkiller:

```text
MemFree:        124052532 kB
MemAvailable:   123345084 kB
SwapCached:            0 kB
SwapTotal:      16777212 kB
SwapFree:       16777212 kB
AnonPages:        147856 kB
Mapped:            85088 kB
Slab:             619672 kB
SUnreclaim:       531840 kB
```

Post-prune top RSS is now dominated by expected worker-host services:

```text
dockerd            85 MB
containerd         32 MB
multipathd         22 MB
systemd-journald   22 MB
NetworkManager     13 MB
systemd --user     12 MB
polkitd             7 MB
```

Post-prune service state:

```text
gnome-remote-desktop inactive/masked
snapd                inactive/masked
fwupd                inactive/masked
packagekit           inactive/masked
apport               inactive/masked
upower               inactive/masked
udisks2              inactive/masked
switcheroo-control   inactive/masked
rtkit-daemon         inactive/masked
NetworkManager       active/enabled
wpa_supplicant       active/enabled
docker               active/enabled
containerd           active/enabled
polkit               active/static
systemd-journald     active/static
```

## Remaining validation

This fixes the obvious soulkiller host overhead and stale swap state. The definitive validation is to rerun the GLM-5.2 MTP1/DCP4/128K live swap test and compare GPU-worker `VmSwap`, `majflt`, `pswpin`, and `pswpout` during a 512-token decode.

## Live validation after prune

Two live-swap probes were run after pruning soulkiller. Each probe launched GLM, waited for readiness, ran one warmup request, then captured direct GPU-worker and global VM counters immediately before and after a measured 512-token decode. These direct snapshots are the authoritative evidence.

The background sampler in these two runs still had a remote PID quoting issue and should not be used for conclusions from these runs. `run_live_swap_probe.sh` has since been fixed for future runs.

### Production profile: DCP4 / 128K / MTP1

Output directory:

```text
logs/live-swap-20260627-142224-prod-mtp1-128k
```

Capacity/perf:

```text
GPU KV cache size: 132,096 tokens
Maximum concurrency for 131,072 tokens/request: 1.01x
warmup_primes_128: 9.625 tok/s
measured_codegen_512: 14.838 tok/s
```

Measured decode deltas:

| Host | GPU-worker VmSwap delta | GPU-worker major-fault delta | pswpin delta | pswpout delta | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| relic | +38,084 kB | +532 | +14,825 pages | +49,812 pages | noisiest node; Codex/tooling host |
| soulkiller | -140 kB | +26 | +1,537 pages | 0 pages | fixed; no longer standout |
| cynosure | -216 kB | +29 | +8,240 pages | +5 pages | okay |
| blackwall | -1,620 kB | +345 | +2,110 pages | 0 pages | moderate faults, no swap-out |

### Experimental profile: DCP4 / 120K / MTP2

Output directory:

```text
logs/live-swap-20260627-144518-exp-mtp2-120k
```

Capacity/perf:

```text
GPU KV cache size: 122,558 tokens
Maximum concurrency for 120,000 tokens/request: 1.02x
warmup_primes_128: 4.862 tok/s
measured_codegen_512: 15.847 tok/s
```

Measured decode deltas:

| Host | GPU-worker VmSwap delta | GPU-worker major-fault delta | pswpin delta | pswpout delta | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| relic | -3,560 kB | +770 | +3,965 pages | 0 pages | noisiest node |
| soulkiller | -152 kB | +19 | +365 pages | 0 pages | fixed; best/near-best behavior |
| cynosure | -248 kB | +25 | +476 pages | +1 page | okay |
| blackwall | -172 kB | +23 | +435 pages | 0 pages | okay |

### Conclusion

Soulkiller is no longer the pathological swap/fault node. Before pruning, soulkiller's GPU worker incurred +1,636 major faults and the host emitted +81,524 `pswpout` pages during one 512-token decode. After pruning:

- MTP1/128K: soulkiller GPU worker +26 major faults, 0 `pswpout`.
- MTP2/120K: soulkiller GPU worker +19 major faults, 0 `pswpout`.

The remaining concern is relic, not soulkiller. Relic hosts Codex and several Node/MCP processes, and in the production MTP1/128K run it was the only node that swapped out during the measured decode window.
