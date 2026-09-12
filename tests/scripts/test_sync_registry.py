"""Tests for sync_registry.py script."""

import importlib.util
import os
import tomllib

import pytest


@pytest.fixture
def sync_module():
    """Import sync_registry.py as a module."""
    script_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "sync_registry.py"
    )
    spec = importlib.util.spec_from_file_location("sync_registry", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSyncRegistry:
    def test_plugin_source_override_keeps_other_plugins_on_base_url(self, tmp_path, sync_module):
        for name, extra in [("upstream", ""), ("custom", 'source = "https://example.com/fork/plugin.toml"\n')]:
            directory = tmp_path / name
            directory.mkdir()
            (directory / "plugin.toml").write_text(f'[plugin]\nid = "example.{name}"\n{extra}')
        output = tmp_path / "registry.toml"
        sync_module.generate_registry(str(tmp_path), str(output), "https://example.com/plugins")
        entries = {entry["id"]: entry["source"] for entry in tomllib.loads(output.read_text())["plugins"]}
        assert entries == {
            "example.custom": "https://example.com/fork/plugin.toml",
            "example.upstream": "https://example.com/plugins/upstream/plugin.toml",
        }

    def test_generates_registry_from_plugin_toml(self, tmp_path, sync_module):
        """Registry is generated from plugin.toml files."""
        plugin_dir = tmp_path / "plugins" / "my_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.toml").write_text(
            '[plugin]\n'
            'id = "com.example.my-plugin"\n'
            'name = "My Plugin"\n'
            'description = "A test"\n'
            'version = "1.0.0"\n'
            'author = "Alice"\n'
            'min_wenzi_version = "0.1.0"\n'
            'files = ["__init__.py"]\n'
        )
        registry_path = tmp_path / "plugins" / "registry.toml"
        sync_module.generate_registry(
            plugins_dir=str(tmp_path / "plugins"),
            output_path=str(registry_path),
            base_url="https://raw.githubusercontent.com/Airead/WenZi/refs/heads/main/plugins",
        )
        content = registry_path.read_text()
        assert 'name = "WenZi Official"' in content
        assert 'id = "com.example.my-plugin"' in content
        assert 'name = "My Plugin"' in content
        assert "source = " in content

    def test_skips_dirs_without_plugin_toml(self, tmp_path, sync_module):
        """Directories without plugin.toml are skipped."""
        (tmp_path / "plugins" / "no_toml").mkdir(parents=True)
        registry_path = tmp_path / "plugins" / "registry.toml"
        sync_module.generate_registry(
            plugins_dir=str(tmp_path / "plugins"),
            output_path=str(registry_path),
            base_url="https://example.com/plugins",
        )
        content = registry_path.read_text()
        assert "[[plugins]]" not in content

    def test_skips_dirs_without_id(self, tmp_path, sync_module):
        """Plugins without id field are skipped with a warning."""
        plugin_dir = tmp_path / "plugins" / "no_id"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.toml").write_text(
            '[plugin]\nname = "No ID"\n'
        )
        registry_path = tmp_path / "plugins" / "registry.toml"
        sync_module.generate_registry(
            plugins_dir=str(tmp_path / "plugins"),
            output_path=str(registry_path),
            base_url="https://example.com/plugins",
        )
        content = registry_path.read_text()
        assert "[[plugins]]" not in content
