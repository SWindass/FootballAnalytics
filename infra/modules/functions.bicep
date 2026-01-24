// Azure Functions module (Consumption plan)

@description('Name of the function app')
param name string

@description('Location for the app')
param location string

@description('Tags to apply')
param tags object

@description('Key Vault name')
param keyVaultName string

@description('PostgreSQL connection string')
@secure()
param postgresConnectionString string

// Storage account for functions
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: replace('${name}st', '-', '')
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

// App Service Plan (Consumption)
resource hostingPlan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: '${name}-plan'
  location: location
  tags: tags
  sku: {
    name: 'Y1'
    tier: 'Dynamic'
  }
  properties: {
    reserved: true  // Linux
  }
}

// Function App
resource functionApp 'Microsoft.Web/sites@2023-01-01' = {
  name: name
  location: location
  tags: tags
  kind: 'functionapp,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: hostingPlan.id
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.11'
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'DATABASE_URL_SYNC'
          value: postgresConnectionString
        }
        {
          name: 'FOOTBALL_DATA_API_KEY'
          value: '@Microsoft.KeyVault(VaultName=${keyVaultName};SecretName=football-data-api-key)'
        }
        {
          name: 'ODDS_API_KEY'
          value: '@Microsoft.KeyVault(VaultName=${keyVaultName};SecretName=odds-api-key)'
        }
        {
          name: 'ANTHROPIC_API_KEY'
          value: '@Microsoft.KeyVault(VaultName=${keyVaultName};SecretName=anthropic-api-key)'
        }
      ]
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
    }
    httpsOnly: true
  }
}

// Key Vault access for function app
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource keyVaultRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, functionApp.id, 'Key Vault Secrets User')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: functionApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output name string = functionApp.name
output id string = functionApp.id
output defaultHostName string = functionApp.properties.defaultHostName
