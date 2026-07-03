##
# Register Gym environments.
##

try:
    from isaaclab_tasks.utils import import_packages
except ModuleNotFoundError:
    # FDM dataset utilities can run without IsaacLab.
    import_packages = None

if import_packages is not None:
    # The blacklist is used to prevent importing configs from sub-packages
    _BLACKLIST_PKGS = ["virtual_obstacle"]
    # Import all configs in this package
    import_packages(__name__, _BLACKLIST_PKGS)
