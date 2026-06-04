use crate::errors::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProviderKind {
    Cpu,
    Metal,
    Cuda,
}

impl ExecutionProviderKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "metal" => Some(Self::Metal),
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderCompatibilityClass {
    CpuPortable,
    MetalFastPath,
    CudaFastPath,
    HeterogeneousPortable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryModel {
    SystemRam,
    DiscreteVram,
    UnifiedMemory,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendContractDescriptor {
    pub provider: ExecutionProviderKind,
    pub compatibility_class: ProviderCompatibilityClass,
    pub optimization_profile: String,
    pub supports_decode_microbatch: bool,
    pub supports_paged_kv: bool,
    pub supports_checkpoint_handoff: bool,
    pub supports_device_sampling: bool,
    pub fast_path_eligible: bool,
    pub memory_model: MemoryModel,
    pub contract_hash: String,
}

impl BackendContractDescriptor {
    pub fn for_provider(provider: ExecutionProviderKind) -> Self {
        let compatibility_class = match provider {
            ExecutionProviderKind::Cpu => ProviderCompatibilityClass::CpuPortable,
            ExecutionProviderKind::Metal => ProviderCompatibilityClass::MetalFastPath,
            ExecutionProviderKind::Cuda => ProviderCompatibilityClass::CudaFastPath,
        };
        let optimization_profile = match provider {
            ExecutionProviderKind::Cpu => "cpu_serial",
            ExecutionProviderKind::Metal => "metal_vectorized",
            ExecutionProviderKind::Cuda => "cuda_fused",
        }
        .to_string();
        let supports_decode_microbatch = !matches!(provider, ExecutionProviderKind::Cpu);
        let supports_paged_kv = !matches!(provider, ExecutionProviderKind::Cpu);
        let supports_checkpoint_handoff = true;
        let supports_device_sampling = !matches!(provider, ExecutionProviderKind::Cpu);
        let fast_path_eligible = !matches!(provider, ExecutionProviderKind::Cpu);
        let memory_model = match provider {
            ExecutionProviderKind::Cpu => MemoryModel::SystemRam,
            ExecutionProviderKind::Metal => MemoryModel::UnifiedMemory,
            ExecutionProviderKind::Cuda => MemoryModel::DiscreteVram,
        };
        let mut descriptor = Self {
            provider,
            compatibility_class,
            optimization_profile,
            supports_decode_microbatch,
            supports_paged_kv,
            supports_checkpoint_handoff,
            supports_device_sampling,
            fast_path_eligible,
            memory_model,
            contract_hash: String::new(),
        };
        descriptor.contract_hash = descriptor.compute_contract_hash();
        descriptor
    }

    fn compute_contract_hash(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.provider.hash(&mut hasher);
        self.compatibility_class.hash(&mut hasher);
        self.optimization_profile.hash(&mut hasher);
        self.supports_decode_microbatch.hash(&mut hasher);
        self.supports_paged_kv.hash(&mut hasher);
        self.supports_checkpoint_handoff.hash(&mut hasher);
        self.supports_device_sampling.hash(&mut hasher);
        self.fast_path_eligible.hash(&mut hasher);
        self.memory_model.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProviderInfo {
    pub kind: ExecutionProviderKind,
    pub available: bool,
    pub reason: Option<String>,
    pub contract: BackendContractDescriptor,
}

pub fn detect_execution_providers() -> Vec<ExecutionProviderInfo> {
    let mut providers = vec![ExecutionProviderInfo {
        kind: ExecutionProviderKind::Cpu,
        available: true,
        reason: None,
        contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Cpu),
    }];

    #[cfg(target_os = "macos")]
    {
        providers.push(ExecutionProviderInfo {
            kind: ExecutionProviderKind::Metal,
            available: cfg!(target_arch = "aarch64"),
            reason: if cfg!(target_arch = "aarch64") {
                None
            } else {
                Some("metal provider requires Apple Silicon for production support".to_string())
            },
            contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Metal),
        });
    }

    #[cfg(not(target_os = "macos"))]
    {
        providers.push(ExecutionProviderInfo {
            kind: ExecutionProviderKind::Metal,
            available: false,
            reason: Some("metal provider is only available on macOS".to_string()),
            contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Metal),
        });
    }

    #[cfg(target_os = "linux")]
    {
        providers.push(ExecutionProviderInfo {
            kind: ExecutionProviderKind::Cuda,
            available: true,
            reason: None,
            contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Cuda),
        });
    }

    #[cfg(not(target_os = "linux"))]
    {
        providers.push(ExecutionProviderInfo {
            kind: ExecutionProviderKind::Cuda,
            available: false,
            reason: Some("cuda provider is only available on Linux builds".to_string()),
            contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Cuda),
        });
    }

    providers
}

pub fn default_execution_contract(
    providers: &[ExecutionProviderInfo],
) -> BackendContractDescriptor {
    providers
        .iter()
        .find(|provider| provider.available && provider.kind != ExecutionProviderKind::Cpu)
        .map(|provider| provider.contract.clone())
        .unwrap_or_else(|| BackendContractDescriptor::for_provider(ExecutionProviderKind::Cpu))
}

pub fn resolve_requested_contract(
    requested_contract_hash: Option<&str>,
    providers: &[ExecutionProviderInfo],
) -> Result<BackendContractDescriptor> {
    let descriptor = match requested_contract_hash {
        Some(contract_hash) => providers
            .iter()
            .find(|provider| provider.contract.contract_hash == contract_hash),
        None => providers.iter().find(|provider| {
            provider.contract.contract_hash == default_execution_contract(providers).contract_hash
        }),
    }
    .ok_or_else(|| {
        let selected = requested_contract_hash.unwrap_or("<default>");
        AgentError::Config(format!(
            "Backend contract {} is not described on this node",
            selected
        ))
    })?;

    if !descriptor.available {
        return Err(AgentError::Config(format!(
            "Backend contract {} is unavailable: {}",
            descriptor.contract.contract_hash,
            descriptor
                .reason
                .clone()
                .unwrap_or_else(|| "no reason provided".to_string())
        )));
    }

    Ok(descriptor.contract.clone())
}

static SELECTED_BACKEND_CONTRACT: OnceLock<BackendContractDescriptor> = OnceLock::new();

pub fn set_selected_backend_contract(contract: BackendContractDescriptor) -> Result<()> {
    match SELECTED_BACKEND_CONTRACT.set(contract.clone()) {
        Ok(()) => Ok(()),
        Err(existing) if existing == contract => Ok(()),
        Err(existing) => Err(AgentError::Config(format!(
            "Backend contract already initialized to {}, cannot change to {}",
            existing.contract_hash, contract.contract_hash
        ))),
    }
}

pub fn selected_backend_contract() -> Option<BackendContractDescriptor> {
    SELECTED_BACKEND_CONTRACT.get().cloned()
}
