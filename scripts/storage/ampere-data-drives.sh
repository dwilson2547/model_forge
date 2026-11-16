#!/bin/bash
# One-time setup of the Altra's two 4 TB data drives (2026-11-15): wipe, format ext4, mount at
# /mnt/data1 and /mnt/data2. Run on the Altra:   sudo bash ampere-data-drives.sh
# The drives are addressed by serial, never by nvmeN (that numbering changed between boots).
# It refuses to run if a drive is missing, mounted, or is the disk holding /.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
OWNER=${SUDO_USER:-daniel}
declare -A DRIVE=(
  [data1]=/dev/disk/by-id/nvme-INTEL_SSDPE2KX040T7_PHLF720401524P0IGN
  [data2]=/dev/disk/by-id/nvme-INTEL_SSDPE2KX040T7_PHLF720401384P0IGN
)
ROOTDISK=$(lsblk -no PKNAME "$(findmnt -no SOURCE /)")

echo "This will ERASE these two drives:"
for l in data1 data2; do
  d=${DRIVE[$l]}
  [ -b "$d" ] || { echo "  $l: $d not found - stopping, nothing changed"; exit 1; }
  real=$(basename "$(readlink -f "$d")")
  [ "$real" != "$ROOTDISK" ] || { echo "  $l: $d is the boot disk - stopping"; exit 1; }
  if lsblk -no MOUNTPOINT "/dev/$real" | grep -q .; then
    echo "  $l: /dev/$real has something mounted - stopping, nothing changed"; exit 1; fi
  echo "  $l -> /mnt/$l : /dev/$real  $(lsblk -dno MODEL,SERIAL,SIZE "/dev/$real")"
done
echo "The boot disk ($ROOTDISK) is not touched."
read -r -p "Type ERASE to continue: " ok
[ "$ok" = ERASE ] || { echo "cancelled, nothing changed"; exit 1; }

for l in data1 data2; do
  d=${DRIVE[$l]}
  for p in "$d"-part*; do [ -b "$p" ] && wipefs -aq "$p"; done
  wipefs -aq "$d"
  mkfs.ext4 -q -m 0 -L "$l" "$d"          # -m 0: no 5% root reserve on a data volume
  mkdir -p "/mnt/$l"
  grep -q "^LABEL=$l " /etc/fstab || echo "LABEL=$l  /mnt/$l  ext4  noatime,nofail  0  2" >> /etc/fstab
done
systemctl daemon-reload
udevadm settle
mount -a
for l in data1 data2; do
  mountpoint -q "/mnt/$l" || { echo "/mnt/$l did not mount - check 'journalctl -b | tail'"; exit 1; }
  chown "$OWNER:$OWNER" "/mnt/$l"
done
echo "Done:"
df -h /mnt/data1 /mnt/data2
