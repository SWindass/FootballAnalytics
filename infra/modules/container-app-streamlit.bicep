// Azure Container Apps module for Streamlit Dashboard

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

@description('Container Apps Environment ID')
param containerAppEnvId string

@description('PostgreSQL connection string')
@secure()
param postgresConnectionString string

// Get reference to existing container registry
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: containerRegistryName
}

// Container App for Streamlit
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: containerAppEnvId
    configuration: {
      ingress: {
        external: true
        targetPort: 8501
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
          name: 'database-url'
          value: postgresConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'streamlit'
          image: containerImage
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          env: [
            {
              name: 'DATABASE_URL_SYNC'
              secretRef: 'database-url'
            }
            {
              name: 'ENVIRONMENT'
              value: 'production'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0  // Scale to zero when not in use (free!)
        maxReplicas: 2
        rules: [
          {
            name: 'http-rule'
            http: {
              metadata: {
                concurrentRequests: '20'
              }
            }
          }
        ]
      }
    }
  }
}

output name string = containerApp.name
output url string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
