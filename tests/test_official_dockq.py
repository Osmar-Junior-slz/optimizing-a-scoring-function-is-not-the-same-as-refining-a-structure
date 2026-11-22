from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from docking_confidence.evaluation import official_dockq as subject


def _pdb(path, chains=("A", "B")):
    lines=[]
    serial=1
    for chain,y in zip(chains,(0.,4.5)):
        for residue in range(1,4):
            for atom,dx,dy,dz in (("N",0,0,0),("CA",1,0,0),("C",1,1,0),("O",0,1,0),("CB",1,0,1)):
                x=(residue-1)*3.8+dx
                lines.append(f"ATOM  {serial:5d} {atom:^4s} ALA {chain}{residue:4d}    {x:8.3f}{y+dy:8.3f}{dz:8.3f}{1.:6.2f}{20.:6.2f}          {atom[0]:>2s}")
                serial+=1
    path.write_text("\n".join(lines)+"\nTER\nEND\n",encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path):
    return _pdb(tmp_path/"model.pdb"),_pdb(tmp_path/"native.pdb")


def _fake_response(**changes):
    result={"available":True,"implementation":"DockQ","implementation_version":"2.1.3",
        "implementation_sha256":"a"*64,"constants_sha256":"b"*64,"python_executable":"isolated-python",
        "metrics":{"DockQ":.8,"fnat":.7,"iRMSD":.2,"LRMSD":.3},"official_interface_result":{"DockQ":.8}}
    result.update(changes)
    return result


def test_metric_contract_rejects_peptide_and_noncanonical_numeric_types():
    descriptor=subject.quality_metric_descriptor()
    assert descriptor["contact_cutoff_angstrom"]==5 and descriptor["interface_cutoff_angstrom"]==10
    assert subject.quality_metric_sha256(descriptor)=="77cd3f01b02de14f482a00110598d04bbcc4f8beacaf84d4b92076b1798a8b1d"
    for changes in ({"capri_peptide":True},{"contact_cutoff_angstrom":4},{"interface_cutoff_angstrom":8}, {"contact_cutoff_angstrom":5.0}):
        with pytest.raises(ValueError):subject.quality_metric_sha256({**descriptor,**changes})


def test_standard_result_binds_hashes_mapping_and_version(inputs,tmp_path,monkeypatch):
    model,native=inputs;seen={}
    def backend(request,**runtime):
        seen.update(request)
        assert runtime["dependency_path"]=="isolated"
        return _fake_response()
    monkeypatch.setattr(subject,"_invoke_backend",backend)
    result=subject.evaluate_standard_complex(model,native,receptor_chain="A",ligand_chain="B",output_dir=tmp_path/"out",dependency_path="isolated")
    assert seen["native_to_model_chain_map"]=={"A":"A","B":"B"}
    assert seen["expected_version"]=="2.1.3"
    assert result["quality_metric_id"]==subject.METRIC_ID
    assert result["dockq"]==.8 and result["metrics"]["dockq_category"]=="high"
    assert result["inputs"]["model_sha256"]==hashlib.sha256(model.read_bytes()).hexdigest()
    saved=json.loads(Path(result["metrics_path"]).read_bytes());digest=saved.pop("record_sha256")
    assert subject.canonical_sha256(saved)==digest
    with pytest.raises(FileExistsError):
        subject.evaluate_standard_complex(model,native,receptor_chain="A",ligand_chain="B",output_dir=tmp_path/"out")


def test_explicit_reference_mapping_is_not_optimized(inputs,tmp_path,monkeypatch):
    model,_=inputs;native=_pdb(tmp_path/"different_ids.pdb",("C","D"));seen={}
    def backend(request,**runtime):seen.update(request);return _fake_response()
    monkeypatch.setattr(subject,"_invoke_backend",backend)
    subject.evaluate_standard_complex(model,native,receptor_chain="A",ligand_chain="B",reference_receptor_chain="C",reference_ligand_chain="D",output_dir=tmp_path/"out")
    assert seen["native_to_model_chain_map"]=={"C":"A","D":"B"}


def test_missing_official_evaluator_never_falls_back(inputs,tmp_path,monkeypatch):
    def unavailable(*args,**kwargs):raise subject.OfficialDockQUnavailableError("missing DockQ")
    monkeypatch.setattr(subject,"_invoke_backend",unavailable)
    assert subject.official_dockq_preflight()["available"] is False
    with pytest.raises(subject.OfficialDockQUnavailableError):
        subject.evaluate_standard_complex(*inputs,receptor_chain="A",ligand_chain="B",output_dir=tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_bad_hash_version_or_mutating_input_does_not_write_result(inputs,tmp_path,monkeypatch):
    model,native=inputs
    monkeypatch.setattr(subject,"_invoke_backend",lambda *args,**kwargs:_fake_response())
    with pytest.raises(ValueError,match="Reference"):
        subject.evaluate_standard_complex(model,native,receptor_chain="A",ligand_chain="B",output_dir=tmp_path/"hash",expected_reference_sha256="f"*64)
    monkeypatch.setattr(subject,"_invoke_backend",lambda *args,**kwargs:_fake_response(implementation_version="2.1.4"))
    with pytest.raises(ValueError,match="frozen metric"):
        subject.evaluate_standard_complex(model,native,receptor_chain="A",ligand_chain="B",output_dir=tmp_path/"version")
    def mutate(*args,**kwargs):model.write_text(model.read_text()+"REMARK changed\n");return _fake_response()
    monkeypatch.setattr(subject,"_invoke_backend",mutate)
    with pytest.raises(ValueError,match="changed"):
        subject.evaluate_standard_complex(model,native,receptor_chain="A",ligand_chain="B",output_dir=tmp_path/"mutation")
    assert not any((tmp_path/name).exists() for name in ("hash","version","mutation"))


@pytest.mark.parametrize("changes",[{"DockQ":float("nan")},{"DockQ":1.2},{"fnat":-1},{"iRMSD":-1}])
def test_invalid_official_values_are_rejected(inputs,tmp_path,monkeypatch,changes):
    response=_fake_response();response["metrics"].update(changes)
    monkeypatch.setattr(subject,"_invoke_backend",lambda *args,**kwargs:response)
    with pytest.raises(ValueError,match="invalid metrics"):
        subject.evaluate_standard_complex(*inputs,receptor_chain="A",ligand_chain="B",output_dir=tmp_path/"out")


def test_actual_official_perfect_pose_when_isolated_runtime_present(inputs,tmp_path):
    dependency=Path(__file__).resolve().parents[1]/".local/dockq_standard_runtime"
    if not dependency.exists():pytest.skip("Optional isolated DockQ runtime is not installed")
    preflight=subject.official_dockq_preflight(dependency_path=dependency)
    if not preflight["available"]:pytest.skip(preflight["error"])
    result=subject.evaluate_standard_complex(*inputs,receptor_chain="A",ligand_chain="B",output_dir=tmp_path/"official",dependency_path=dependency)
    assert result["dockq"]==pytest.approx(1,abs=1e-10)
    assert result["fnat"]==pytest.approx(1)
    assert result["irmsd"]==pytest.approx(0,abs=1e-6)
    assert result["quality_metric"]["capri_peptide"] is False
