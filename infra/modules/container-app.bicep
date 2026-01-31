// Azure Container Apps module

@description('Name of the container app')
param name string

@description('Location for the app')
param location string

@description('Tags to apply')
param tags object

@description('Container registry name')
param containerRegistryName string

@description('Container image')
param containerImage string

@description('Key Vault name')
param keyVaultName string

@description('PostgreSQL host')
param postgresHost string

@description('PostgreSQL database name')
param postgresDatabase string

@description('PostgreSQL username')
param postgresUser string

// Get references to existing resources
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: containerRegistryName
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

// Container Apps Environment (Consumption plan)
resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: '${name}-env'
  location: location
  tags: tags
  properties: {
    zoneRedundant: false
  }
}

// Container App
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
        {
          name: 'postgres-password'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/postgres-password'
          identity: 'system'
        }
        {
          name: 'football-data-api-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/football-data-api-key'
          identity: 'system'
        }
        {
          name: 'odds-api-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/odds-api-key'
          identity: 'system'
        }
        {
          name: 'anthropic-api-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/anthropic-api-key'
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImage
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          env: [
            {
              name: 'ENVIRONMENT'
              value: 'production'
            }
            {
              name: 'DATABASE_URL'
              value: 'postgresql+asyncpg://${postgresUser}:$(postgres-password)@${postgresHost}:5432/${postgresDatabase}'
            }
            {
              name: 'DATABASE_URL_SYNC'
              value: 'postgresql://${postgresUser}:$(postgres-password)@${postgresHost}:5432/${postgresDatabase}'
            }
            {
              name: 'FOOTBALL_DATA_API_KEY'
              secretRef: 'football-data-api-key'
            }
            {
              name: 'ODDS_API_KEY'
              secretRef: 'odds-api-key'
            }
            {
              name: 'ANTHROPIC_API_KEY'
              secretRef: 'anthropic-api-key'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 3
        rules: [
          {
            name: 'http-rule'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}

// Grant Key Vault access to container app
resource keyVaultRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, containerApp.id, 'Key Vault Secrets User')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output name string = containerApp.name
output url string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output id string = containerApp.id
output environmentId string = containerAppEnv.id
