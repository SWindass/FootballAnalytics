// Azure Key Vault module

@description('Name of the key vault')
param name string

@description('Location for the vault')
param location string

@description('Tags to apply')
param tags object

@description('Secrets to store')
param secrets array = []

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    publicNetworkAccess: 'Enabled'
  }
}

// Store secrets
resource secretResources 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = [for secret in secrets: {
  parent: keyVault
  name: secret.name
  properties: {
    value: secret.value
  }
}]

output name string = keyVault.name
output id string = keyVault.id
output uri string = keyVault.properties.vaultUri
